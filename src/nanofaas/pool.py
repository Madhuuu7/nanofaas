"""The warm pool: who gets a container, who waits, and who starts one.

This is where a function platform is won or lost. Everything else is
plumbing; this file decides what happens when fifty requests arrive at a
function that is scaled to zero.

## The stampede

A naive pool answers "no warm container? start one" per request. Fifty
simultaneous requests therefore start fifty containers, on a machine that can
usefully start about two at a time. Each start is slower because they contend,
so every one of the fifty requests waits longer than it would have if it had
simply queued behind a single start -- and at the end there are forty-nine
containers nobody needed, holding memory until the idle reaper takes them.

``max_concurrent_starts`` caps how many containers may be starting at once.
The requests that cannot start one wait on a condition variable instead, and
take whichever container becomes ready first. ``scripts/bench.py`` runs both
policies against the same burst and reports the difference, which is the
measurement this project is built around.

The cap is not "start exactly one". A function with ``concurrency: 1`` and
sustained load genuinely needs more containers, so the pool keeps starting
them -- just never more than ``max_concurrent_starts`` at a time, and never
more than ``max_instances`` in total.

## Scale to zero

A container with no work is a container costing memory. The reaper stops any
instance idle for longer than its function's ``idle_timeout_s``, and when the
last one goes the function is at zero and the next request pays a cold start.
That trade -- idle cost against cold-start probability -- has an optimum that
depends on the traffic, and ``policy.py`` computes it rather than guessing.

## Why a condition variable and not a queue of requests

Waiting threads need to re-evaluate *the whole situation* when anything
changes: a slot freeing, a container becoming ready, a start failing. A
request queue would have to decide who to wake and for what; a condition
broadcast lets each waiter re-check the predicate it actually cares about,
which is "is there capacity for me now", and that predicate is the same one
the admission path uses. One rule, one place.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .functions import FunctionSpec
from .metrics import ContainerEvent, Invocation, Metrics
from .runtime import ColdStartFailed, ContainerRuntime, Instance

log = logging.getLogger("nanofaas.pool")


class CapacityTimeout(RuntimeError):
    """No slot became available before the caller's deadline."""


class FunctionFailed(RuntimeError):
    """The function itself failed: a bad handler, not a platform problem."""


@dataclass
class PoolConfig:
    #: How many containers may be starting at once, across the whole
    #: platform. Two, because this is a laptop and starts contend for the
    #: same daemon; a bigger host can afford more. Setting it to a very large
    #: number reproduces the naive stampede, which is how the benchmark gets
    #: its comparison.
    max_concurrent_starts: int = 2
    #: How long a request waits for capacity before giving up. Separate from
    #: the function's own execution timeout: this is queueing, not running.
    acquire_timeout_s: float = 30.0
    #: How often the reaper looks for idle containers.
    reap_interval_s: float = 1.0
    #: Keep this many containers warm per function regardless of idleness.
    #: Zero is the point of the project; anything else is buying latency.
    min_instances: int = 0


@dataclass
class FunctionState:
    """Per-function pool state, guarded by the pool's lock."""

    spec: FunctionSpec
    instances: list[Instance] = field(default_factory=list)
    starting: int = 0
    #: Set when a start fails, so waiting requests fail fast with the reason
    #: instead of sitting until their deadline.
    last_error: str = ""
    last_error_at: float = 0.0


class Pool:
    def __init__(
        self,
        runtime: ContainerRuntime,
        metrics: Metrics | None = None,
        config: PoolConfig | None = None,
    ):
        self.runtime = runtime
        self.metrics = metrics or Metrics()
        self.config = config or PoolConfig()
        self._lock = threading.RLock()
        self._available = threading.Condition(self._lock)
        self._states: dict[str, FunctionState] = {}
        self._global_starting = 0
        self._stop = threading.Event()
        self._reaper: threading.Thread | None = None

    # -- lifecycle --------------------------------------------------------

    def start_reaper(self) -> None:
        if self._reaper is not None:
            return
        self._reaper = threading.Thread(
            target=self._reap_loop, name="nanofaas-reaper", daemon=True
        )
        self._reaper.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._reaper is not None:
            self._reaper.join(timeout=5)
        with self._lock:
            states = list(self._states.values())
        for state in states:
            for instance in list(state.instances):
                self._retire(state, instance, reason="shutdown")

    # -- admission --------------------------------------------------------

    def _state(self, spec: FunctionSpec) -> FunctionState:
        state = self._states.get(spec.name)
        if state is None:
            state = FunctionState(spec=spec)
            self._states[spec.name] = state
        else:
            # A redeploy replaces the spec; existing containers belong to the
            # old image and must not serve the new function.
            if state.spec.digest() != spec.digest():
                for instance in list(state.instances):
                    self._retire(state, instance, reason="redeployed")
            state.spec = spec
        return state

    def _free_slot(self, state: FunctionState) -> Instance | None:
        """A healthy instance with room, least-loaded first.

        Least-loaded rather than round-robin because concurrency above one
        means instances differ in how busy they are, and handing a request to
        an idle container instead of a busy one is the entire decision
        available at this point.
        """
        candidates = [
            instance
            for instance in state.instances
            if not instance.poisoned
            and not instance.stopped
            and instance.in_flight < state.spec.concurrency
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda i: i.in_flight)

    def _may_start(self, state: FunctionState) -> bool:
        total = len(state.instances) + state.starting
        return (
            total < state.spec.max_instances
            and self._global_starting < self.config.max_concurrent_starts
        )

    def acquire(self, spec: FunctionSpec, timeout_s: float | None = None) -> tuple[Instance, float, float]:
        """Get a slot on a container, starting one if that is the right move.

        Returns ``(instance, queued_ms, cold_start_ms)``. ``cold_start_ms`` is
        non-zero only for the request that actually paid for the start -- the
        requests that queued behind it waited, but they did not cause it, and
        attributing the cold start to all of them would overstate how often
        the platform pays one.
        """
        deadline = time.time() + (
            timeout_s if timeout_s is not None else self.config.acquire_timeout_s
        )
        waited_from = time.perf_counter()

        with self._lock:
            state = self._state(spec)
            while True:
                instance = self._free_slot(state)
                if instance is not None:
                    instance.in_flight += 1
                    instance.last_used_at = time.time()
                    queued_ms = (time.perf_counter() - waited_from) * 1000.0
                    return instance, queued_ms, 0.0

                if self._may_start(state):
                    state.starting += 1
                    self._global_starting += 1
                    break  # start it outside the lock

                # A recent start failure is this request's failure too: the
                # next attempt would hit the same broken image.
                if state.last_error and time.time() - state.last_error_at < 2.0:
                    raise FunctionFailed(state.last_error)

                remaining = deadline - time.time()
                if remaining <= 0:
                    raise CapacityTimeout(
                        f"{spec.name}: no capacity after "
                        f"{(time.perf_counter() - waited_from):.1f}s "
                        f"({len(state.instances)} instances, "
                        f"{state.starting} starting, "
                        f"max {spec.max_instances})"
                    )
                self._available.wait(timeout=min(remaining, 0.5))

        # -- outside the lock: the slow part ------------------------------
        try:
            instance = self.runtime.start(spec)
        except ColdStartFailed as exc:
            with self._lock:
                state.starting -= 1
                self._global_starting -= 1
                state.last_error = str(exc)
                state.last_error_at = time.time()
                self._available.notify_all()
            raise FunctionFailed(str(exc)) from exc
        except BaseException:
            with self._lock:
                state.starting -= 1
                self._global_starting -= 1
                self._available.notify_all()
            raise

        with self._lock:
            state.starting -= 1
            self._global_starting -= 1
            state.last_error = ""
            state.instances.append(instance)
            instance.in_flight += 1
            instance.last_used_at = time.time()
            # Other waiters may now fit on this container, or may be able to
            # start one of their own now that a start slot is free.
            self._available.notify_all()

        self.metrics.record_event(
            ContainerEvent(
                function=spec.name,
                instance=instance.name,
                kind="start",
                at=time.time(),
                cold_start_ms=instance.cold_start.total_ms,
            )
        )
        queued_ms = (
            time.perf_counter() - waited_from
        ) * 1000.0 - instance.cold_start.total_ms
        return instance, max(queued_ms, 0.0), instance.cold_start.total_ms

    def release(self, instance: Instance, *, poisoned: bool = False) -> None:
        with self._lock:
            instance.in_flight = max(0, instance.in_flight - 1)
            instance.last_used_at = time.time()
            instance.invocations += 1
            if poisoned:
                instance.poisoned = True
            self._available.notify_all()
        if poisoned:
            state = self._states.get(instance.function)
            if state is not None and instance.in_flight == 0:
                self._retire(state, instance, reason="poisoned")

    # -- the request path -------------------------------------------------

    def invoke(
        self, spec: FunctionSpec, event: dict[str, Any], *, timeout_s: float | None = None
    ) -> dict[str, Any]:
        """Acquire, call, release, record."""
        overall = time.perf_counter()
        instance, queued_ms, cold_ms = self.acquire(spec)
        poisoned = False
        status = 500
        exec_ms = 0.0
        error = ""
        body: dict[str, Any] = {}
        try:
            status, body = self.runtime.invoke(
                instance, event, timeout_s or spec.timeout_s
            )
            exec_ms = float(body.get("exec_ms") or 0.0)
            if status >= 500:
                error = str(body.get("error") or "")[:300]
        except Exception as exc:  # noqa: BLE001 - httpx timeouts and friends
            # A timed-out invocation may have left the shim still running it,
            # so this container cannot safely serve the next request.
            poisoned = True
            status = 504
            error = f"{type(exc).__name__}: {exc}"[:300]
            body = {"error": "the invocation timed out or the container failed"}
        finally:
            self.release(instance, poisoned=poisoned)

        total_ms = (time.perf_counter() - overall) * 1000.0
        self.metrics.record(
            Invocation(
                function=spec.name,
                at=time.time(),
                queued_ms=round(queued_ms, 3),
                cold_start_ms=round(cold_ms, 3),
                exec_ms=round(exec_ms, 3),
                total_ms=round(total_ms, 3),
                cold=cold_ms > 0,
                status=status,
                instance=instance.name,
                error=error,
            )
        )
        return {
            "status": status,
            "body": body,
            "cold": cold_ms > 0,
            "cold_start_ms": round(cold_ms, 2),
            "queued_ms": round(queued_ms, 2),
            "exec_ms": round(exec_ms, 3),
            "total_ms": round(total_ms, 2),
            "instance": instance.name,
        }

    # -- scale to zero ----------------------------------------------------

    def _retire(self, state: FunctionState, instance: Instance, *, reason: str) -> None:
        with self._lock:
            if instance not in state.instances:
                return
            state.instances.remove(instance)
            lifetime = instance.age_s
            invocations = instance.invocations
        self.runtime.stop(instance, force=reason in ("poisoned", "shutdown"))
        self.metrics.record_event(
            ContainerEvent(
                function=state.spec.name,
                instance=instance.name,
                kind="stop",
                at=time.time(),
                reason=reason,
                lifetime_s=round(lifetime, 2),
            )
        )
        log.debug(
            "retired %s (%s) after %.1fs and %d invocations",
            instance.name, reason, lifetime, invocations,
        )

    def reap_idle(self) -> int:
        """Stop containers idle past their function's timeout."""
        retired = 0
        with self._lock:
            plan: list[tuple[FunctionState, Instance]] = []
            for state in self._states.values():
                keep = max(self.config.min_instances, 0)
                idle = [
                    instance
                    for instance in state.instances
                    if instance.in_flight == 0
                    and instance.idle_s >= state.spec.idle_timeout_s
                ]
                # Oldest first: a long-lived container has had more chances
                # to accumulate whatever a leak accumulates.
                idle.sort(key=lambda i: i.started_at)
                spare = len(state.instances) - keep
                for instance in idle[: max(spare, 0)]:
                    plan.append((state, instance))
        for state, instance in plan:
            self._retire(state, instance, reason="idle")
            retired += 1
        if retired:
            with self._lock:
                self._available.notify_all()
        return retired

    def _reap_loop(self) -> None:
        while not self._stop.wait(self.config.reap_interval_s):
            try:
                self.reap_idle()
            except Exception:  # noqa: BLE001 - the reaper must not die
                log.exception("reaper")

    def scale_to_zero(self, name: str) -> int:
        with self._lock:
            state = self._states.get(name)
            instances = list(state.instances) if state else []
        for instance in instances:
            self._retire(state, instance, reason="manual")
        return len(instances)

    # -- introspection ----------------------------------------------------

    def instances(self, name: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            states = (
                [self._states[name]] if name and name in self._states
                else list(self._states.values())
            )
            return [
                instance.to_dict() for state in states for instance in state.instances
            ]

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "functions": {
                    name: {
                        "instances": len(state.instances),
                        "starting": state.starting,
                        "in_flight": sum(i.in_flight for i in state.instances),
                        "capacity": sum(
                            max(state.spec.concurrency - i.in_flight, 0)
                            for i in state.instances
                            if not i.poisoned
                        ),
                        "idle_timeout_s": state.spec.idle_timeout_s,
                        "max_instances": state.spec.max_instances,
                    }
                    for name, state in self._states.items()
                },
                "starting_now": self._global_starting,
                "max_concurrent_starts": self.config.max_concurrent_starts,
            }
