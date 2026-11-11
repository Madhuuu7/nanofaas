"""A container runtime that starts instantly and never leaves the process.

Kept out of conftest so test modules can import it by name: ``tests`` is
deliberately not a package -- adding an __init__.py there changes how pytest
resolves rootdir and imports -- so a relative import from conftest does not
work, and reaching into conftest for a helper is a confusing thing to read
anyway.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from nanofaas.functions import FunctionSpec
from nanofaas.runtime import ColdStart, ColdStartFailed, Instance


@dataclass
class FakeRuntime:
    """A container runtime that starts instantly and never leaves the process.

    ``start_delay_s`` exists so tests can make a start slow enough to observe
    the pool's behaviour *during* one -- which is the whole subject of the
    single-flight and queueing tests -- without waiting three real seconds.
    """

    start_delay_s: float = 0.02
    fail_starts: bool = False
    fail_message: str = "the handler failed to import"
    #: Per-invocation behaviour the tests drive.
    exec_delay_s: float = 0.0
    invoke_status: int = 200
    invoke_raises: bool = False

    started: list[str] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)
    live: int = 0
    max_live: int = 0
    concurrent_starts: int = 0
    max_concurrent_starts_seen: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _counter: int = 0

    def start(self, spec: FunctionSpec) -> Instance:
        with self._lock:
            self._counter += 1
            name = f"fake-{spec.name}-{self._counter}"
            self.concurrent_starts += 1
            self.max_concurrent_starts_seen = max(
                self.max_concurrent_starts_seen, self.concurrent_starts
            )
        try:
            if self.start_delay_s:
                time.sleep(self.start_delay_s)
            if self.fail_starts:
                raise ColdStartFailed(f"{spec.name}: {self.fail_message}")
        finally:
            with self._lock:
                self.concurrent_starts -= 1

        with self._lock:
            self.started.append(name)
            self.live += 1
            self.max_live = max(self.max_live, self.live)

        instance = Instance(name=name, function=spec.name, container=None)
        instance.host_port = 9000 + self._counter
        instance.cold_start = ColdStart(
            create_ms=1.0, start_ms=1.0, port_ms=0.5,
            ready_ms=self.start_delay_s * 1000.0, probes=1,
        )
        return instance

    def invoke(self, instance: Instance, event, timeout_s: float):
        if self.invoke_raises:
            raise TimeoutError("the invocation timed out")
        if self.exec_delay_s:
            time.sleep(self.exec_delay_s)
        if self.invoke_status >= 500:
            return self.invoke_status, {"error": "the handler raised", "exec_ms": 0.1}
        return self.invoke_status, {"result": {"echo": event}, "exec_ms": 0.1}

    def stop(self, instance: Instance, *, force: bool = False) -> None:
        with self._lock:
            if not instance.stopped or instance.name not in self.stopped:
                self.stopped.append(instance.name)
                self.live = max(0, self.live - 1)
        instance.stopped = True

    def logs(self, instance: Instance, tail: int = 50) -> str:
        return "(fake)"

    def close(self) -> None:
        return None

    def reap_orphans(self) -> int:
        return 0


# -- a platform with no daemon behind it -----------------------------------
#
# The console recorder drives the real control plane over HTTP. To exercise
# that script -- and the replay contract the published page depends on --
# without a Docker daemon, the same app is wired to a runtime that emulates
# the example handlers instead of running them. It is a stand-in for the
# container, so emulating the handler is exactly its job: `fib` has to refuse
# the argument the real one refuses, or a recording of it proves nothing.


class ConsoleRuntime(FakeRuntime):
    """``FakeRuntime``, but it answers like the example functions do."""

    def invoke(self, instance: Instance, event, timeout_s: float):
        if self.exec_delay_s:
            time.sleep(self.exec_delay_s)
        name = instance.function
        if name == "fib":
            n = int(event.get("n", 25))
            if n > 32:
                return 500, {
                    "error": "ValueError: n too large for a demo function",
                    "traceback": (
                        'Traceback (most recent call last):\n'
                        '  File "/app/handler.py", line 4, in handle\n'
                        "    raise ValueError('n too large for a demo function')\n"
                        "ValueError: n too large for a demo function"
                    ),
                    "exec_ms": 0.4,
                }
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            return 200, {"result": {"n": n, "iterative": a}, "exec_ms": 1.8}
        return 200, {
            "result": {"greeting": f"hello {event.get('name', 'world')}"},
            "exec_ms": 0.3,
        }


@dataclass
class FakeBuild:
    tag: str
    cached: bool = False
    build_ms: float = 1620.0
    size_mb: float = 58.4


@contextmanager
def faked_platform(**runtime_kwargs):
    """Make ``create_app`` come up ready, with no daemon anywhere near it.

    Patches the two module-level functions that reach for Docker and the
    platform's own ``connect``, then puts them back. Used by the recorder's
    tests and by nothing that ships.
    """
    import importlib

    from nanofaas.pool import Pool, PoolConfig

    # By name: ``nanofaas.api.app`` is also the FastAPI instance the package
    # re-exports, and `from nanofaas.api import app` hands back that object
    # rather than the module the patches have to land on.
    app_module = importlib.import_module("nanofaas.api.app")

    runtime_kwargs.setdefault("start_delay_s", 0.05)
    built: list[str] = []

    def connect(self) -> None:
        self.docker = object()
        self.runtime = ConsoleRuntime(**runtime_kwargs)
        self.pool = Pool(self.runtime, self.metrics, PoolConfig(max_concurrent_starts=2))
        self.pool.start_reaper()
        self.error = None

    def build_image(_client, spec):
        cached = spec.name in built
        built.append(spec.name)
        return FakeBuild(tag=f"nanofaas/{spec.name}:fake", cached=cached)

    originals = (app_module.Platform.connect, app_module.build_image,
                 app_module.remove_image, app_module.ensure_base_images)
    app_module.Platform.connect = connect
    app_module.build_image = build_image
    app_module.remove_image = lambda *_a, **_k: None
    app_module.ensure_base_images = lambda *_a, **_k: None
    try:
        yield
    finally:
        (app_module.Platform.connect, app_module.build_image,
         app_module.remove_image, app_module.ensure_base_images) = originals
