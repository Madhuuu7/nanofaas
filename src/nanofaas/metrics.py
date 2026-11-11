"""What happened, kept in a ring buffer.

Latency is reported as percentiles and never as a mean. A mean invocation
time on a scale-to-zero platform is a number with no referent: the
distribution is two separate populations -- a few cold starts near three
seconds and thousands of warm calls near four milliseconds -- and their
average describes neither. One cold start in a thousand moves the mean by
3ms, which is most of the warm median, so a mean that moved would be
indistinguishable from a real regression.

So invocations are stored individually, in a bounded ring, and summarised by
quantile and by kind. Bounded because this runs on a laptop and an unbounded
list of every invocation is a memory leak with a progress bar.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Invocation:
    """One request's journey, in milliseconds."""

    function: str
    at: float
    #: Waiting for a slot: either for a container to start, or behind other
    #: requests on a busy one. The number scale-to-zero is judged on.
    queued_ms: float
    #: Non-zero only when this request caused a container to start.
    cold_start_ms: float
    #: The handler's own time, as the shim measured it.
    exec_ms: float
    #: Everything, from the platform receiving the request to returning it.
    total_ms: float
    cold: bool
    status: int
    instance: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def quantile(values: list[float], q: float) -> float:
    """Nearest-rank percentile.

    Not interpolated: with a bimodal cold/warm distribution, interpolating
    between the 999th and 1000th sample invents a latency that sits in the
    empty gap between the two populations and that no request experienced.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if q <= 0:
        return ordered[0]
    if q >= 1:
        return ordered[-1]
    import math

    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


def summarise(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "p50": round(quantile(values, 0.50), 2),
        "p90": round(quantile(values, 0.90), 2),
        "p95": round(quantile(values, 0.95), 2),
        "p99": round(quantile(values, 0.99), 2),
        "max": round(max(values), 2),
        "min": round(min(values), 2),
    }


@dataclass
class ContainerEvent:
    """A container starting or stopping, for the lifecycle timeline."""

    function: str
    instance: str
    kind: str  # "start" | "stop"
    at: float
    reason: str = ""
    cold_start_ms: float = 0.0
    lifetime_s: float = 0.0
    #: The phase breakdown of the start this container paid for, kept on the
    #: event rather than only on the live container. A platform at zero has
    #: no containers to ask, and "where did the cold start go" is a question
    #: about starts that have happened, not about ones still running.
    phases: dict[str, float] | None = None


class Metrics:
    """A bounded history of invocations and container lifecycle events."""

    def __init__(self, capacity: int = 20_000):
        self._lock = threading.Lock()
        self._invocations: deque[Invocation] = deque(maxlen=capacity)
        self._events: deque[ContainerEvent] = deque(maxlen=2_000)
        self.started_at = time.time()

    def record(self, invocation: Invocation) -> None:
        with self._lock:
            self._invocations.append(invocation)

    def record_event(self, event: ContainerEvent) -> None:
        with self._lock:
            self._events.append(event)

    def invocations(self, function: str | None = None, limit: int = 0) -> list[Invocation]:
        with self._lock:
            rows = [
                row
                for row in self._invocations
                if function is None or row.function == function
            ]
        return rows[-limit:] if limit else rows

    def events(self, limit: int = 200) -> list[ContainerEvent]:
        with self._lock:
            return list(self._events)[-limit:]

    def summary(self, function: str | None = None) -> dict[str, Any]:
        rows = self.invocations(function)
        if not rows:
            return {"invocations": 0, "cold_starts": 0}
        cold = [row for row in rows if row.cold]
        warm = [row for row in rows if not row.cold]
        ok = [row for row in rows if 200 <= row.status < 300]
        return {
            "invocations": len(rows),
            "cold_starts": len(cold),
            "cold_start_rate": round(len(cold) / len(rows), 4),
            "errors": len(rows) - len(ok),
            "total_ms": summarise([row.total_ms for row in rows]),
            # The split that makes the numbers mean something: these are two
            # populations, and the headline percentile of the mixture is
            # mostly a statement about the cold-start rate.
            "warm_ms": summarise([row.total_ms for row in warm]),
            # And the split inside the warm population, which a burst makes
            # the difference between a true sentence and a false one. A warm
            # request that waited for a busy container paid for capacity, not
            # for a container start; counting that wait as the cost of a warm
            # invocation makes a burst look like warm calls are slow, when
            # what happened is that they were fast and there were not enough
            # containers to run them on.
            "warm_service_ms": summarise(
                [max(row.total_ms - row.queued_ms, 0.0) for row in warm]
            ),
            "warm_queued": sum(1 for row in warm if row.queued_ms >= 1.0),
            "cold_ms": summarise([row.total_ms for row in cold]),
            "queued_ms": summarise([row.queued_ms for row in rows]),
            "exec_ms": summarise([row.exec_ms for row in rows if row.exec_ms > 0]),
        }

    def by_function(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            names = sorted({row.function for row in self._invocations})
        return {name: self.summary(name) for name in names}

    def reset(self) -> None:
        with self._lock:
            self._invocations.clear()
            self._events.clear()
            self.started_at = time.time()
