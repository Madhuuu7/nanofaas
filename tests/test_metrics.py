"""The metrics ring, and why it never reports a mean.

The distribution of invocation latencies on a scale-to-zero platform is two
populations three orders of magnitude apart: cold starts near three seconds,
warm calls near four milliseconds. Their average describes neither, and one
cold start in a thousand shifts it by about as much as the entire warm median
-- so a mean that moved would be indistinguishable from a real regression.
These tests pin that the summary keeps them separate and that the percentiles
are the nearest-rank kind rather than interpolated.
"""

from __future__ import annotations

import time

import pytest

from nanofaas.metrics import ContainerEvent, Invocation, Metrics, quantile, summarise


def inv(total_ms: float, *, cold: bool = False, function: str = "hello",
        status: int = 200, queued_ms: float = 0.0, exec_ms: float = 1.0) -> Invocation:
    return Invocation(
        function=function,
        at=time.time(),
        queued_ms=queued_ms,
        cold_start_ms=total_ms if cold else 0.0,
        exec_ms=exec_ms,
        total_ms=total_ms,
        cold=cold,
        status=status,
    )


# -- percentiles -----------------------------------------------------------


def test_quantiles_are_nearest_rank_not_interpolated():
    """With a bimodal distribution, interpolating between the 999th and
    1000th sample invents a latency in the empty gap between the two
    populations -- a number no request experienced."""
    values = [1.0] * 99 + [3000.0]
    assert quantile(values, 0.99) == 1.0
    assert quantile(values, 0.995) == 3000.0
    # An interpolating implementation would return something around 1500 here.
    assert quantile(values, 0.99) in (1.0, 3000.0)


def test_quantile_edges():
    values = [5.0, 1.0, 9.0]
    assert quantile(values, 0.0) == 1.0
    assert quantile(values, 1.0) == 9.0
    assert quantile(values, -1) == 1.0
    assert quantile(values, 2) == 9.0
    assert quantile([], 0.5) == 0.0


def test_summarise_reports_the_shape_not_a_mean():
    summary = summarise([1.0, 2.0, 3.0, 4.0, 100.0])
    assert summary["n"] == 5
    assert summary["min"] == 1.0
    assert summary["max"] == 100.0
    assert summary["p50"] == 3.0
    assert "mean" not in summary


def test_summarise_of_nothing_is_not_a_crash():
    assert summarise([]) == {"n": 0}


# -- the two populations ---------------------------------------------------


def test_cold_and_warm_are_summarised_separately():
    metrics = Metrics()
    for _ in range(99):
        metrics.record(inv(4.0))
    metrics.record(inv(3200.0, cold=True))

    summary = metrics.summary()
    assert summary["invocations"] == 100
    assert summary["cold_starts"] == 1
    assert summary["cold_start_rate"] == 0.01
    assert summary["warm_ms"]["n"] == 99
    assert summary["warm_ms"]["p50"] == 4.0
    assert summary["cold_ms"]["n"] == 1
    assert summary["cold_ms"]["p50"] == 3200.0


def test_a_warm_call_that_queued_is_not_charged_to_the_warm_population():
    """Found on the published console, which said a warm call was as slow as
    a cold start.

    A burst of ten at a function scaled to zero starts a few containers and
    makes the rest wait for one. Those waiters never paid for a start, so
    they are warm -- but their total is almost all queueing, and summarising
    it as the cost of a warm invocation inverts what happened: the warm path
    was fast and there were not enough containers to run on. Queueing is a
    capacity number and belongs apart from both populations.
    """
    metrics = Metrics()
    for _ in range(3):
        metrics.record(inv(880.0, cold=True))
    for _ in range(7):
        metrics.record(inv(860.0, queued_ms=859.0))

    summary = metrics.summary()
    assert summary["warm_ms"]["p50"] == 860.0, "the total is still reported"
    assert summary["warm_service_ms"]["p50"] == 1.0, "and what it cost to serve"
    assert summary["warm_queued"] == 7
    assert summary["cold_ms"]["p50"] == 880.0


def test_warm_service_time_equals_the_total_when_nothing_waits():
    metrics = Metrics()
    for _ in range(10):
        metrics.record(inv(4.0))
    summary = metrics.summary()
    assert summary["warm_service_ms"]["p50"] == summary["warm_ms"]["p50"] == 4.0
    assert summary["warm_queued"] == 0


def test_the_mixture_percentile_mostly_reports_the_cold_start_rate():
    """The point the console makes in prose, asserted here: at a 1% cold rate
    the p99 of the mixture is a cold start, so it is a statement about how
    often containers start rather than about latency."""
    metrics = Metrics()
    for _ in range(99):
        metrics.record(inv(4.0))
    metrics.record(inv(3200.0, cold=True))
    assert metrics.summary()["total_ms"]["p99"] == 4.0
    assert metrics.summary()["total_ms"]["max"] == 3200.0


def test_errors_are_counted_separately_from_latency():
    metrics = Metrics()
    metrics.record(inv(4.0))
    metrics.record(inv(9.0, status=500))
    metrics.record(inv(12.0, status=504))
    summary = metrics.summary()
    assert summary["errors"] == 2
    assert summary["invocations"] == 3


def test_an_empty_history_summarises_to_zero():
    assert Metrics().summary() == {"invocations": 0, "cold_starts": 0}


# -- the ring --------------------------------------------------------------


def test_the_history_is_bounded():
    """Unbounded, this is a memory leak with a progress bar: the platform
    runs for days on a laptop."""
    metrics = Metrics(capacity=100)
    for i in range(500):
        metrics.record(inv(float(i)))
    rows = metrics.invocations()
    assert len(rows) == 100
    assert rows[0].total_ms == 400.0  # the oldest 400 were dropped
    assert rows[-1].total_ms == 499.0


def test_filtering_and_limiting_by_function():
    metrics = Metrics()
    for i in range(5):
        metrics.record(inv(float(i), function="alpha"))
    for i in range(3):
        metrics.record(inv(float(i), function="beta"))
    assert len(metrics.invocations("alpha")) == 5
    assert len(metrics.invocations("beta")) == 3
    assert len(metrics.invocations("alpha", limit=2)) == 2
    assert metrics.invocations("nope") == []


def test_by_function_summarises_each_separately():
    metrics = Metrics()
    metrics.record(inv(4.0, function="alpha"))
    metrics.record(inv(3000.0, cold=True, function="beta"))
    by_function = metrics.by_function()
    assert set(by_function) == {"alpha", "beta"}
    assert by_function["alpha"]["cold_starts"] == 0
    assert by_function["beta"]["cold_starts"] == 1


def test_container_events_are_kept_for_the_timeline():
    metrics = Metrics()
    metrics.record_event(
        ContainerEvent(function="hello", instance="a", kind="start",
                       at=time.time(), cold_start_ms=3200.0)
    )
    metrics.record_event(
        ContainerEvent(function="hello", instance="a", kind="stop",
                       at=time.time(), reason="idle", lifetime_s=31.0)
    )
    events = metrics.events()
    assert [e.kind for e in events] == ["start", "stop"]
    assert events[1].reason == "idle"


def test_reset_clears_everything():
    metrics = Metrics()
    metrics.record(inv(4.0))
    metrics.record_event(ContainerEvent("hello", "a", "start", time.time()))
    metrics.reset()
    assert metrics.summary()["invocations"] == 0
    assert metrics.events() == []


def test_an_invocation_serialises_for_the_api():
    row = inv(4.0, queued_ms=1.5).to_dict()
    assert row["total_ms"] == 4.0
    assert row["queued_ms"] == 1.5
    assert row["cold"] is False
    assert set(row) >= {"function", "at", "status", "exec_ms"}


def test_recording_is_safe_from_several_threads():
    """The pool records from whichever worker thread served the request, so
    a dropped or torn record here would be a silent undercount."""
    import threading

    metrics = Metrics(capacity=5000)

    def worker():
        for _ in range(200):
            metrics.record(inv(4.0))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert metrics.summary()["invocations"] == 1600
