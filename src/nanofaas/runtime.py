"""Container lifecycle, and where a cold start actually goes.

A cold start is not one number, it is four, and knowing which one dominates
is the difference between optimising something and guessing:

``create_ms``   the daemon writes a container config and a writable layer
``start_ms``    the process is forked, namespaced and cgrouped
``port_ms``     the published host port is read back from the daemon
``ready_ms``    the interpreter boots, imports the handler, answers a probe

``ready_ms`` is usually the largest and it is the only one a function author
can change -- it is their imports. ``create_ms`` and ``start_ms`` belong to
the platform and the daemon, and no amount of application tuning moves them,
which is exactly why they are reported separately rather than summed.

## Why the port is published on loopback only

``ports={"8080/tcp": ("127.0.0.1", 0)}`` -- the empty port asks the daemon for
an ephemeral one, and the explicit ``127.0.0.1`` is the part that matters.
Published without a host address, Docker binds ``0.0.0.0`` and every function
on the platform is reachable from anything else on the network, which on a
laptop joined to a cafe wifi is a genuinely bad afternoon.

## What the container cannot do

Memory and CPU are hard caps, the root filesystem is read-only with a small
tmpfs at /tmp, every capability is dropped, ``no-new-privileges`` is set, the
pid count is capped, and the process is a non-root user from the image.

What it *can* still do is reach the network, because the platform talks to it
over HTTP and a published port needs a bridge. Blocking egress properly means
an internal network and a different transport, and claiming otherwise would
be the kind of security story that reads well and is false. More to the
point: a container shares the host kernel. This is a robustness boundary
against a function that misbehaves, not a sandbox against one that is
hostile. The README says so plainly.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .functions import FunctionSpec

log = logging.getLogger("nanofaas.runtime")

#: The port the shim listens on inside the container.
CONTAINER_PORT = 8080

#: How often to poll the readiness endpoint. 5ms is far below the time a
#: container takes to boot, so the probe interval does not measurably pad the
#: cold-start figure -- at 50ms it would add up to 25ms of pure rounding to
#: every measurement.
PROBE_INTERVAL_S = 0.005

#: Give up on a container that never becomes ready. Longer than any plausible
#: interpreter boot, short enough that a broken image does not hang a deploy.
READY_TIMEOUT_S = 30.0


class ColdStartFailed(RuntimeError):
    """A container was created but never became ready."""


@dataclass
class ColdStart:
    """Where the time went, in milliseconds."""

    create_ms: float = 0.0
    start_ms: float = 0.0
    port_ms: float = 0.0
    ready_ms: float = 0.0
    probes: int = 0

    @property
    def total_ms(self) -> float:
        return self.create_ms + self.start_ms + self.port_ms + self.ready_ms

    def to_dict(self) -> dict[str, float]:
        return {
            "create_ms": round(self.create_ms, 2),
            "start_ms": round(self.start_ms, 2),
            "port_ms": round(self.port_ms, 2),
            "ready_ms": round(self.ready_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "probes": self.probes,
        }


@dataclass
class Instance:
    """One running container, and what the pool knows about it."""

    name: str
    function: str
    container: Any = field(repr=False, default=None)
    host_port: int = 0
    cold_start: ColdStart = field(default_factory=ColdStart)
    started_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    invocations: int = 0
    #: In-flight requests. The pool reads this to decide whether an instance
    #: has room, so it is guarded by the pool's lock, not this object's.
    in_flight: int = 0
    #: Set when an invocation times out: the shim may still be stuck on it,
    #: so the instance is retired rather than handed to the next caller.
    poisoned: bool = False
    stopped: bool = False

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.host_port}"

    @property
    def idle_s(self) -> float:
        return time.time() - self.last_used_at

    @property
    def age_s(self) -> float:
        return time.time() - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "function": self.function,
            "host_port": self.host_port,
            "cold_start": self.cold_start.to_dict(),
            "age_s": round(self.age_s, 2),
            "idle_s": round(self.idle_s, 2),
            "invocations": self.invocations,
            "in_flight": self.in_flight,
            "poisoned": self.poisoned,
        }


class ContainerRuntime:
    """Starts, probes, stops and reaps function containers."""

    def __init__(self, client, *, label: str = "nanofaas"):
        self.client = client
        self.label = label
        self._counter = 0
        self._counter_lock = threading.Lock()
        # One HTTP client for the life of the runtime, not one per call. A
        # fresh client per invocation re-resolves proxies and opens a new TCP
        # connection to the daemon's published port every time, which cost
        # ~880ms per warm invocation here against ~4ms with the connection
        # pool reused. `trust_env=False` keeps an ambient HTTP_PROXY out of
        # the path to a loopback address, which is never what anyone means.
        self._http = httpx.Client(
            timeout=READY_TIMEOUT_S,
            trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=64, max_connections=128),
        )

    def _next_name(self, spec: FunctionSpec) -> str:
        with self._counter_lock:
            self._counter += 1
            return f"{self.label}-{spec.name}-{self._counter}-{int(time.time() * 1000) % 100000}"

    def start(self, spec: FunctionSpec) -> Instance:
        """Create, start and wait for a container to answer its probe."""
        timing = ColdStart()
        name = self._next_name(spec)

        t0 = time.perf_counter()
        container = self.client.containers.create(
            image=spec.image_tag,
            name=name,
            detach=True,
            # -- resource caps -------------------------------------------
            mem_limit=f"{spec.memory_mb}m",
            # Docker takes CPU as billionths of a core.
            nano_cpus=int(spec.cpus * 1e9),
            pids_limit=128,
            # -- hardening ------------------------------------------------
            read_only=True,
            tmpfs={"/tmp": "rw,noexec,nosuid,size=16m"},
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            # -- wiring ---------------------------------------------------
            ports={f"{CONTAINER_PORT}/tcp": ("127.0.0.1", 0)},
            environment={
                "NANOFAAS_PORT": str(CONTAINER_PORT),
                "NANOFAAS_ENTRYPOINT": spec.entrypoint,
                "NANOFAAS_HANDLER": f"/app/{spec.handler_filename}",
                **spec.env,
            },
            labels={
                "nanofaas.function": spec.name,
                "nanofaas.managed": "true",
            },
        )
        timing.create_ms = (time.perf_counter() - t0) * 1000.0

        instance = Instance(name=name, function=spec.name, container=container)
        try:
            t1 = time.perf_counter()
            container.start()
            timing.start_ms = (time.perf_counter() - t1) * 1000.0

            t2 = time.perf_counter()
            instance.host_port = self._host_port(container)
            timing.port_ms = (time.perf_counter() - t2) * 1000.0

            t3 = time.perf_counter()
            timing.probes = self._wait_ready(instance, spec)
            timing.ready_ms = (time.perf_counter() - t3) * 1000.0
        except BaseException:
            # A half-started container is worse than none: it holds a port
            # and a name, and the name collision is what makes the *next*
            # start fail too.
            self.stop(instance, force=True)
            raise

        instance.cold_start = timing
        instance.started_at = time.time()
        instance.last_used_at = instance.started_at
        return instance

    def _host_port(self, container) -> int:
        """Read back the ephemeral port the daemon assigned.

        Retried because the port map is populated asynchronously: a reload
        immediately after start can legitimately return an empty mapping, and
        treating that as a failure makes cold starts flaky at random.
        """
        deadline = time.time() + 10.0
        while time.time() < deadline:
            container.reload()
            ports = (container.attrs.get("NetworkSettings") or {}).get("Ports") or {}
            binding = ports.get(f"{CONTAINER_PORT}/tcp")
            if binding:
                return int(binding[0]["HostPort"])
            time.sleep(PROBE_INTERVAL_S)
        raise ColdStartFailed(
            f"{container.name}: the daemon never published a host port"
        )

    def _wait_ready(self, instance: Instance, spec: FunctionSpec) -> int:
        """Poll until the shim answers, or the container dies trying."""
        probes = 0
        deadline = time.time() + READY_TIMEOUT_S
        url = f"{instance.url}/health"
        while time.time() < deadline:
            probes += 1
            try:
                response = self._http.get(url, timeout=2.0)
                if response.status_code == 200:
                    return probes
                if response.status_code == 503:
                    # The shim is up and telling us the handler did not
                    # import. Waiting will not fix it, and the traceback
                    # it carries is the useful part.
                    detail = response.json().get("error") or response.text
                    raise ColdStartFailed(
                        f"{spec.name}: the handler failed to import\n{detail}"
                    )
            except httpx.HTTPError:
                pass
            # Cheap liveness check: if the container exited, the probe
            # will never succeed and the remaining 30 seconds are wasted.
            if probes % 20 == 0 and not self._is_running(instance):
                raise ColdStartFailed(
                    f"{spec.name}: the container exited before becoming ready\n"
                    f"{self.logs(instance, tail=20)}"
                )
            time.sleep(PROBE_INTERVAL_S)
        raise ColdStartFailed(
            f"{spec.name}: not ready after {READY_TIMEOUT_S:.0f}s "
            f"({probes} probes)\n{self.logs(instance, tail=20)}"
        )

    def _is_running(self, instance: Instance) -> bool:
        try:
            instance.container.reload()
            return instance.container.status == "running"
        except Exception:  # noqa: BLE001
            return False

    def invoke(
        self, instance: Instance, event: dict[str, Any], timeout_s: float
    ) -> tuple[int, dict[str, Any]]:
        """One HTTP call into a warm container."""
        response = self._http.post(
            f"{instance.url}/invoke", json=event, timeout=timeout_s
        )
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, {"error": "the shim returned a non-JSON body"}

    def logs(self, instance: Instance, tail: int = 50) -> str:
        try:
            return instance.container.logs(tail=tail).decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            return f"(could not read logs: {exc})"

    def stop(self, instance: Instance, *, force: bool = False) -> None:
        """Stop and remove a container.

        ``force`` kills instead of asking politely. A function that is wedged
        will not answer SIGTERM, and waiting ten seconds per container to
        discover that makes scale-to-zero feel broken.
        """
        instance.stopped = True
        container = instance.container
        if container is None:
            return
        try:
            if force:
                container.remove(force=True)
            else:
                container.stop(timeout=3)
                container.remove()
        except Exception as exc:  # noqa: BLE001
            log.debug("stopping %s: %s", instance.name, exc)

    def close(self) -> None:
        self._http.close()

    def reap_orphans(self) -> int:
        """Remove containers this platform started and then lost track of.

        A crash leaves running containers with no owner, holding memory and
        ports. They are labelled, so they can be found and cleaned up on the
        next start -- otherwise a few development crashes quietly exhaust the
        machine.
        """
        removed = 0
        try:
            containers = self.client.containers.list(
                all=True, filters={"label": "nanofaas.managed=true"}
            )
        except Exception:  # noqa: BLE001
            return 0
        for container in containers:
            try:
                container.remove(force=True)
                removed += 1
            except Exception:  # noqa: BLE001
                continue
        return removed
