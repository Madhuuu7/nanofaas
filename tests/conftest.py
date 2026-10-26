"""Fixtures, and the line between unit tests and the ones that need Docker.

Most of this platform can be tested without a container runtime, and it is
worth the effort to keep it that way: the pool's admission logic -- who waits,
who starts a container, who is refused -- is the part most likely to be wrong
and the part a real Docker test would exercise slowest and least
deterministically. A 3-second cold start times out a test suite long before it
covers the fourth interleaving.

So ``FakeRuntime`` stands in for ``ContainerRuntime`` with the same surface and
a configurable, instant "start". Tests that genuinely need a daemon are marked
``@pytest.mark.docker`` and skip when there is none, which keeps the suite
green on a fresh clone and in CI without Docker.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import pytest

from nanofaas.functions import EXAMPLES, FunctionSpec
from nanofaas.runtime import ColdStart, ColdStartFailed, Instance


def docker_available() -> bool:
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:  # noqa: BLE001
        return False


HAS_DOCKER = docker_available()

needs_docker = pytest.mark.skipif(
    not HAS_DOCKER, reason="no Docker daemon; start Docker Desktop to run these"
)


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


@pytest.fixture
def spec() -> FunctionSpec:
    return FunctionSpec(
        name="hello",
        source=EXAMPLES["hello"]["source"],
        idle_timeout_s=60.0,
        max_instances=4,
        concurrency=1,
    )


@pytest.fixture
def fake_runtime() -> FakeRuntime:
    return FakeRuntime()
