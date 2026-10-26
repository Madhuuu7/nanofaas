"""Choosing the idle timeout, which is the only real decision in scale-to-zero.

Everything else about a scale-to-zero platform follows from one number: how
long a container waits for more work before it is stopped. Too short and
every request pays a three-second cold start. Too long and you are renting
memory for containers doing nothing. Platforms generally ship a default --
five minutes, fifteen minutes -- and the default is a guess about someone
else's traffic.

It does not have to be a guess, because for any given arrival trace the
answer is exactly computable.

## The model

With one instance and a timeout ``T``, a request is warm if and only if the
gap since the previous request is at most ``T``. So, walking the inter-arrival
gaps of a real trace:

* ``gap > T``   -- the container had already stopped: a cold start, and the
  platform paid ``T`` seconds of idle before stopping it
* ``gap <= T``  -- the container was still warm: no cold start, and the
  platform paid ``gap`` seconds of idle

That is not an approximation of the policy, it *is* the policy, so the sweep
below is an exact replay rather than a model of one. What it leaves out is
concurrency: with several instances the warm/cold question depends on which
instance the request lands on. The single-instance answer is the right one
for the low-traffic functions where scale-to-zero actually matters, and the
README says so rather than pretending otherwise.

## The closed form, as a check

For Poisson arrivals at rate ``lambda``, gaps are exponential, so

    P(cold) = P(gap > T) = exp(-lambda * T)

and the timeout that holds the cold-start rate at ``r`` is

    T* = -ln(r) / lambda

``compare_with_closed_form`` runs the simulation against that identity. They
agree to sampling noise, which is how I know the replay is computing what the
algebra says it is -- two independent routes to the same number, rather than
one route and a hope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PolicyResult:
    """What one idle timeout would have cost on a given trace."""

    idle_timeout_s: float
    requests: int
    cold_starts: int
    cold_start_rate: float
    #: Seconds of container life spent idle, waiting for work that may not come.
    idle_seconds: float
    #: Seconds spent actually running the handler.
    busy_seconds: float
    #: What the memory reservation costs, in the unit cloud providers bill in.
    gb_seconds: float
    #: Mean added latency per request from cold starts, in milliseconds.
    mean_latency_penalty_ms: float
    p95_latency_penalty_ms: float

    @property
    def container_seconds(self) -> float:
        return self.idle_seconds + self.busy_seconds

    def to_dict(self) -> dict[str, float]:
        return {
            "idle_timeout_s": self.idle_timeout_s,
            "requests": self.requests,
            "cold_starts": self.cold_starts,
            "cold_start_rate": round(self.cold_start_rate, 5),
            "idle_seconds": round(self.idle_seconds, 2),
            "busy_seconds": round(self.busy_seconds, 2),
            "container_seconds": round(self.container_seconds, 2),
            "gb_seconds": round(self.gb_seconds, 3),
            "mean_latency_penalty_ms": round(self.mean_latency_penalty_ms, 2),
            "p95_latency_penalty_ms": round(self.p95_latency_penalty_ms, 2),
        }


def simulate(
    gaps: Sequence[float],
    idle_timeout_s: float,
    *,
    cold_start_s: float = 3.0,
    exec_s: float = 0.01,
    memory_mb: int = 128,
) -> PolicyResult:
    """Replay a trace of inter-arrival gaps under one idle timeout."""
    gaps = np.asarray(list(gaps), dtype=float)
    requests = int(gaps.size) + 1  # the first request has no preceding gap
    if requests == 1:
        return PolicyResult(
            idle_timeout_s, 1, 1, 1.0, idle_timeout_s, exec_s,
            (idle_timeout_s + exec_s) * memory_mb / 1024.0,
            cold_start_s * 1000.0, cold_start_s * 1000.0,
        )

    cold_mask = gaps > idle_timeout_s
    # The first request always pays a cold start: nothing is warm at t=0.
    cold_starts = int(cold_mask.sum()) + 1
    # Idle time is the gap when warm, and the full timeout when the container
    # was allowed to expire.
    idle_seconds = float(np.where(cold_mask, idle_timeout_s, gaps).sum())
    # The last container also sits idle for a full timeout after the final
    # request, which is easy to forget and is pure cost.
    idle_seconds += idle_timeout_s
    busy_seconds = requests * exec_s

    penalties = np.zeros(requests)
    penalties[0] = cold_start_s * 1000.0
    penalties[1:] = np.where(cold_mask, cold_start_s * 1000.0, 0.0)

    return PolicyResult(
        idle_timeout_s=idle_timeout_s,
        requests=requests,
        cold_starts=cold_starts,
        cold_start_rate=cold_starts / requests,
        idle_seconds=idle_seconds,
        busy_seconds=busy_seconds,
        gb_seconds=(idle_seconds + busy_seconds) * memory_mb / 1024.0,
        mean_latency_penalty_ms=float(penalties.mean()),
        p95_latency_penalty_ms=float(np.quantile(penalties, 0.95)),
    )


def sweep(
    gaps: Sequence[float],
    timeouts: Sequence[float] | None = None,
    **kwargs,
) -> list[PolicyResult]:
    timeouts = timeouts or (0, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900)
    return [simulate(gaps, float(t), **kwargs) for t in timeouts]


def recommend(
    gaps: Sequence[float],
    *,
    target_cold_rate: float = 0.05,
    timeouts: Sequence[float] | None = None,
    **kwargs,
) -> tuple[PolicyResult, str]:
    """The cheapest timeout that holds the cold-start rate under a target.

    Posed this way round on purpose. "Minimise cost plus latency" needs an
    exchange rate between rupees and milliseconds that nobody can state
    honestly, while "keep cold starts under 5% and spend as little as
    possible doing it" is a sentence a team can actually agree to, and the
    answer follows from the trace.
    """
    results = sweep(gaps, timeouts, **kwargs)
    feasible = [r for r in results if r.cold_start_rate <= target_cold_rate]
    if not feasible:
        best = min(results, key=lambda r: r.cold_start_rate)
        return best, (
            f"no timeout in the sweep reaches a {target_cold_rate:.0%} cold-start "
            f"rate: this traffic is too sparse for scale-to-zero to be cheap, and "
            f"the best on offer is {best.cold_start_rate:.1%} at "
            f"{best.idle_timeout_s:g}s. Either accept the cold starts or keep one "
            f"instance warm and stop calling it scale-to-zero."
        )
    best = min(feasible, key=lambda r: r.gb_seconds)
    cheapest = min(results, key=lambda r: r.gb_seconds)
    return best, (
        f"{best.idle_timeout_s:g}s holds cold starts at "
        f"{best.cold_start_rate:.1%} for {best.gb_seconds:.1f} GB-seconds. "
        f"Scaling to zero immediately would cost {cheapest.gb_seconds:.1f} "
        f"GB-seconds and make {cheapest.cold_start_rate:.0%} of requests cold."
    )


def compare_with_closed_form(
    rate_per_minute: float,
    *,
    requests: int = 20_000,
    timeouts: Sequence[float] = (1, 2, 5, 10, 30, 60, 120),
    seed: int = 0,
    **kwargs,
) -> list[dict[str, float]]:
    """Check the replay against the algebra it is supposed to implement.

    For Poisson arrivals the cold-start rate is exactly ``exp(-lambda T)``.
    Simulating the same thing and getting the same answer is how a replay
    earns the right to be trusted on a real trace, where there is no formula
    to check it against.
    """
    lam = rate_per_minute / 60.0
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0 / lam, size=requests - 1)
    rows = []
    for timeout in timeouts:
        measured = simulate(gaps, float(timeout), **kwargs)
        predicted = math.exp(-lam * timeout)
        rows.append(
            {
                "idle_timeout_s": float(timeout),
                "simulated_cold_rate": round(measured.cold_start_rate, 5),
                "closed_form_cold_rate": round(predicted, 5),
                "abs_error": round(abs(measured.cold_start_rate - predicted), 5),
                # Three sigma on a binomial proportion: anything inside this
                # is sampling noise, anything outside is a bug.
                "noise_3sigma": round(
                    3 * math.sqrt(max(predicted * (1 - predicted), 1e-12) / requests), 5
                ),
            }
        )
    return rows


def optimal_timeout_for_poisson(rate_per_minute: float, target_cold_rate: float) -> float:
    """``T* = -ln(r) / lambda``, the closed form, for when a trace is absent."""
    if not 0 < target_cold_rate < 1:
        raise ValueError("target_cold_rate must be in (0, 1)")
    lam = rate_per_minute / 60.0
    if lam <= 0:
        return math.inf
    return -math.log(target_cold_rate) / lam


def gaps_from_timestamps(timestamps: Sequence[float]) -> list[float]:
    """Inter-arrival gaps from absolute times, which is what a log gives you."""
    ordered = sorted(float(t) for t in timestamps)
    return [b - a for a, b in zip(ordered, ordered[1:])]
