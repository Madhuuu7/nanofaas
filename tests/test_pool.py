"""The pool: who gets a container, who waits, and who is refused.

This is the file that matters. The pool decides what happens when a burst
arrives at a function that is scaled to zero, and every interesting case is a
race -- two threads deciding simultaneously whether to start a container, a
slot freeing while someone waits for one, a start failing while others queue
behind it. Against real Docker each of those takes three seconds and resolves
differently every run; against the fake runtime they take twenty milliseconds
and resolve the same way every time, so the concurrency can actually be
pinned rather than hoped at.

The fake runtime counts its own concurrent starts, which is how
``max_concurrent_starts`` is tested for what it claims rather than for what
the pool's bookkeeping believes.
"""

from __future__ import annotations

import threading
import time

import pytest

from nanofaas.functions import EXAMPLES, FunctionSpec
from nanofaas.metrics import Metrics
from nanofaas.pool import CapacityTimeout, FunctionFailed, Pool, PoolConfig

from fakes import FakeRuntime


def make_spec(**overrides) -> FunctionSpec:
    return FunctionSpec(
        **{
            "name": "hello",
            "source": EXAMPLES["hello"]["source"],
            "idle_timeout_s": 60.0,
            "max_instances": 4,
            "concurrency": 1,
            **overrides,
        }
    )


def run_concurrently(fn, n: int) -> list:
    """Fire n threads at once and collect whatever each returns or raises."""
    results: list = [None] * n
    barrier = threading.Barrier(n)

    def worker(i: int) -> None:
        barrier.wait()  # genuinely simultaneous, not merely overlapping
        try:
            results[i] = fn()
        except BaseException as exc:  # noqa: BLE001
            results[i] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    return results


# -- the basic path --------------------------------------------------------


def test_the_first_request_starts_a_container_and_the_second_reuses_it():
    runtime = FakeRuntime()
    pool = Pool(runtime, Metrics())
    spec = make_spec()

    first = pool.invoke(spec, {})
    second = pool.invoke(spec, {})

    assert first["cold"] is True
    assert first["cold_start_ms"] > 0
    assert second["cold"] is False
    assert second["cold_start_ms"] == 0
    assert len(runtime.started) == 1
    assert first["instance"] == second["instance"]
    pool.shutdown()


def test_a_cold_start_is_attributed_only_to_the_request_that_caused_it():
    """The requests that queued behind a start waited, but counting the cold
    start against all of them would overstate how often the platform pays
    one."""
    runtime = FakeRuntime(start_delay_s=0.15)
    pool = Pool(runtime, Metrics())
    spec = make_spec(concurrency=8, max_instances=1)

    results = run_concurrently(lambda: pool.invoke(spec, {}), 6)
    cold = [r for r in results if r["cold"]]
    warm = [r for r in results if not r["cold"]]

    assert len(runtime.started) == 1
    assert len(cold) == 1
    assert len(warm) == 5
    # The ones that waited show it as queue time, not as a cold start.
    assert max(r["queued_ms"] for r in warm) > 0
    pool.shutdown()


def test_the_handlers_failure_is_reported_without_poisoning_the_container():
    """A function that raises is a function bug. The container is still
    perfectly good and must go on serving."""
    runtime = FakeRuntime(invoke_status=500)
    pool = Pool(runtime, Metrics())
    spec = make_spec()

    first = pool.invoke(spec, {})
    second = pool.invoke(spec, {})

    assert first["status"] == 500
    assert second["status"] == 500
    assert len(runtime.started) == 1  # not restarted between the two
    assert runtime.stopped == []
    pool.shutdown()


def test_a_timed_out_invocation_poisons_its_container():
    """The shim may still be running the call, so handing that container to
    the next caller would serve them a process that is already busy."""
    runtime = FakeRuntime(invoke_raises=True)
    pool = Pool(runtime, Metrics())
    spec = make_spec()

    result = pool.invoke(spec, {})
    assert result["status"] == 504
    assert len(runtime.stopped) == 1

    runtime.invoke_raises = False
    recovered = pool.invoke(spec, {})
    assert recovered["status"] == 200
    assert len(runtime.started) == 2  # a fresh container, not the poisoned one
    pool.shutdown()


# -- stampede control ------------------------------------------------------


def test_concurrent_starts_are_capped_at_the_configured_number():
    """Measured inside the fake runtime rather than from the pool's own
    counters, so this tests the behaviour and not the bookkeeping."""
    runtime = FakeRuntime(start_delay_s=0.1)
    pool = Pool(runtime, Metrics(), PoolConfig(max_concurrent_starts=2))
    spec = make_spec(max_instances=8, concurrency=1)

    results = run_concurrently(lambda: pool.invoke(spec, {}), 8)

    assert all(r["status"] == 200 for r in results)
    assert runtime.max_concurrent_starts_seen <= 2
    pool.shutdown()


def test_raising_the_cap_really_does_start_more_at_once():
    """The control is only meaningful if it is the thing deciding; without
    this, a pool that serialised everything for an unrelated reason would
    pass the test above."""
    runtime = FakeRuntime(start_delay_s=0.1)
    pool = Pool(runtime, Metrics(), PoolConfig(max_concurrent_starts=6))
    spec = make_spec(max_instances=8, concurrency=1)

    run_concurrently(lambda: pool.invoke(spec, {}), 8)

    assert runtime.max_concurrent_starts_seen > 2
    pool.shutdown()


def test_queued_requests_reuse_a_container_rather_than_starting_their_own():
    """The point of capping starts: a request that waits takes whichever
    container becomes free first, so a burst of twelve does not leave twelve
    containers behind."""
    runtime = FakeRuntime(start_delay_s=0.08)
    pool = Pool(runtime, Metrics(), PoolConfig(max_concurrent_starts=1))
    spec = make_spec(max_instances=6, concurrency=1)

    results = run_concurrently(lambda: pool.invoke(spec, {}), 12)

    assert all(r["status"] == 200 for r in results)
    assert len(runtime.started) < 12
    pool.shutdown()


def test_the_instance_ceiling_is_never_exceeded():
    """The ceiling on what a traffic spike can cost."""
    runtime = FakeRuntime(start_delay_s=0.05, exec_delay_s=0.05)
    pool = Pool(runtime, Metrics(), PoolConfig(max_concurrent_starts=8))
    spec = make_spec(max_instances=3, concurrency=1)

    run_concurrently(lambda: pool.invoke(spec, {}), 16)

    assert runtime.max_live <= 3
    assert len(pool.instances("hello")) <= 3
    pool.shutdown()


# -- concurrency and queueing ---------------------------------------------


def test_one_container_serves_several_requests_when_concurrency_allows():
    runtime = FakeRuntime(exec_delay_s=0.1)
    pool = Pool(runtime, Metrics())
    spec = make_spec(concurrency=4, max_instances=1)

    results = run_concurrently(lambda: pool.invoke(spec, {}), 4)

    assert all(r["status"] == 200 for r in results)
    assert len(runtime.started) == 1
    pool.shutdown()


def test_concurrency_one_serialises_requests_onto_a_container():
    """With concurrency 1 and a single instance, four requests must take
    about four execution times -- if they overlapped, the limit would not be
    a limit."""
    runtime = FakeRuntime(exec_delay_s=0.08)
    pool = Pool(runtime, Metrics())
    spec = make_spec(concurrency=1, max_instances=1)

    started = time.perf_counter()
    run_concurrently(lambda: pool.invoke(spec, {}), 4)
    elapsed = time.perf_counter() - started

    assert elapsed >= 4 * 0.08 * 0.9
    pool.shutdown()


def test_a_request_that_never_gets_capacity_is_refused_not_hung():
    """A 429 the caller can back off from beats a request that waits
    forever."""
    runtime = FakeRuntime(exec_delay_s=1.5)
    pool = Pool(runtime, Metrics(), PoolConfig(acquire_timeout_s=0.2))
    spec = make_spec(concurrency=1, max_instances=1)

    holder = threading.Thread(target=lambda: pool.invoke(spec, {}), daemon=True)
    holder.start()
    time.sleep(0.3)  # let it take the only slot

    with pytest.raises(CapacityTimeout, match="no capacity"):
        pool.acquire(spec, timeout_s=0.2)
    holder.join(timeout=5)
    pool.shutdown()


def test_a_freed_slot_wakes_a_waiting_request():
    runtime = FakeRuntime(exec_delay_s=0.25)
    pool = Pool(runtime, Metrics())
    spec = make_spec(concurrency=1, max_instances=1)

    results = run_concurrently(lambda: pool.invoke(spec, {}), 3)
    assert all(not isinstance(r, BaseException) for r in results)
    assert all(r["status"] == 200 for r in results)
    assert len(runtime.started) == 1
    pool.shutdown()


# -- failures --------------------------------------------------------------


def test_a_failed_start_is_reported_as_the_functions_fault():
    runtime = FakeRuntime(fail_starts=True)
    pool = Pool(runtime, Metrics())
    spec = make_spec()

    with pytest.raises(FunctionFailed, match="failed to import"):
        pool.invoke(spec, {})
    pool.shutdown()


def test_waiters_fail_fast_after_a_start_fails():
    """Otherwise every queued request sits until its own deadline waiting for
    a container that cannot exist, and a broken deploy looks like a hang."""
    runtime = FakeRuntime(start_delay_s=0.05, fail_starts=True)
    pool = Pool(runtime, Metrics(), PoolConfig(max_concurrent_starts=1,
                                               acquire_timeout_s=10.0))
    spec = make_spec(max_instances=4)

    started = time.perf_counter()
    results = run_concurrently(lambda: pool.invoke(spec, {}), 5)
    elapsed = time.perf_counter() - started

    assert all(isinstance(r, FunctionFailed) for r in results)
    assert elapsed < 5.0  # not five separate ten-second waits
    pool.shutdown()


def test_a_failed_start_releases_its_start_slot():
    """A leaked start slot is the worst kind of bug here: the platform
    silently stops being able to start anything."""
    runtime = FakeRuntime(fail_starts=True)
    pool = Pool(runtime, Metrics(), PoolConfig(max_concurrent_starts=1))
    spec = make_spec()

    for _ in range(3):
        with pytest.raises(FunctionFailed):
            pool.invoke(spec, {})

    runtime.fail_starts = False
    time.sleep(2.1)  # past the fail-fast window
    assert pool.invoke(spec, {})["status"] == 200
    assert pool.status()["starting_now"] == 0
    pool.shutdown()


# -- scale to zero ---------------------------------------------------------


def test_idle_containers_are_reaped_after_their_timeout():
    runtime = FakeRuntime()
    pool = Pool(runtime, Metrics())
    spec = make_spec(idle_timeout_s=0.1)

    pool.invoke(spec, {})
    assert len(pool.instances("hello")) == 1

    time.sleep(0.15)
    assert pool.reap_idle() == 1
    assert pool.instances("hello") == []
    assert len(runtime.stopped) == 1
    pool.shutdown()


def test_a_busy_container_is_never_reaped():
    """Idle time is measured from last use, and a request in flight means it
    is not idle however long it has been running."""
    runtime = FakeRuntime(exec_delay_s=0.4)
    pool = Pool(runtime, Metrics())
    spec = make_spec(idle_timeout_s=0.0)

    worker = threading.Thread(target=lambda: pool.invoke(spec, {}), daemon=True)
    worker.start()
    time.sleep(0.15)

    assert pool.reap_idle() == 0
    assert len(pool.instances("hello")) == 1
    worker.join(timeout=5)
    pool.shutdown()


def test_min_instances_keeps_a_container_warm():
    """The opt-out from scale-to-zero: paying for a container to avoid the
    cold start."""
    runtime = FakeRuntime()
    pool = Pool(runtime, Metrics(), PoolConfig(min_instances=1))
    spec = make_spec(idle_timeout_s=0.0)

    pool.invoke(spec, {})
    time.sleep(0.05)
    assert pool.reap_idle() == 0
    assert len(pool.instances("hello")) == 1
    pool.shutdown()


def test_reaping_is_safe_when_nothing_is_running():
    pool = Pool(FakeRuntime(), Metrics())
    assert pool.reap_idle() == 0
    pool.shutdown()


def test_scale_to_zero_stops_everything_for_one_function():
    runtime = FakeRuntime()
    pool = Pool(runtime, Metrics())
    alpha = make_spec(name="alpha", concurrency=4, max_instances=2)
    beta = make_spec(name="beta")

    pool.invoke(alpha, {})
    pool.invoke(beta, {})
    assert pool.scale_to_zero("alpha") == 1
    assert pool.instances("alpha") == []
    assert len(pool.instances("beta")) == 1
    pool.shutdown()


def test_the_next_request_after_scaling_to_zero_is_cold_again():
    runtime = FakeRuntime()
    pool = Pool(runtime, Metrics())
    spec = make_spec()

    assert pool.invoke(spec, {})["cold"] is True
    assert pool.invoke(spec, {})["cold"] is False
    pool.scale_to_zero("hello")
    assert pool.invoke(spec, {})["cold"] is True
    pool.shutdown()


def test_shutdown_stops_every_container():
    runtime = FakeRuntime()
    pool = Pool(runtime, Metrics())
    pool.invoke(make_spec(name="alpha"), {})
    pool.invoke(make_spec(name="beta"), {})

    pool.shutdown()
    assert runtime.live == 0
    assert len(runtime.stopped) == 2


# -- redeploy --------------------------------------------------------------


def test_a_redeploy_retires_containers_running_the_old_image():
    """They were built from the previous source and would go on serving it,
    which is a deploy that reports success and changes nothing."""
    runtime = FakeRuntime()
    pool = Pool(runtime, Metrics())
    original = make_spec()
    pool.invoke(original, {})
    first_instance = pool.instances("hello")[0]["name"]

    changed = make_spec(source=EXAMPLES["hello"]["source"].replace("world", "there"))
    result = pool.invoke(changed, {})

    assert result["cold"] is True
    assert result["instance"] != first_instance
    assert first_instance in runtime.stopped
    pool.shutdown()


def test_changing_only_a_scaling_knob_keeps_the_container():
    """The digest covers what the image is built from, so a timeout change
    must not throw away a warm container."""
    runtime = FakeRuntime()
    pool = Pool(runtime, Metrics())
    pool.invoke(make_spec(idle_timeout_s=60), {})
    result = pool.invoke(make_spec(idle_timeout_s=5), {})

    assert result["cold"] is False
    assert len(runtime.started) == 1
    pool.shutdown()


# -- reporting -------------------------------------------------------------


def test_status_reports_capacity_and_the_start_cap():
    runtime = FakeRuntime()
    pool = Pool(runtime, Metrics(), PoolConfig(max_concurrent_starts=3))
    pool.invoke(make_spec(concurrency=4), {})

    status = pool.status()
    assert status["max_concurrent_starts"] == 3
    assert status["functions"]["hello"]["instances"] == 1
    assert status["functions"]["hello"]["capacity"] == 4
    assert status["functions"]["hello"]["in_flight"] == 0
    pool.shutdown()


def test_metrics_are_recorded_for_every_invocation():
    metrics = Metrics()
    pool = Pool(FakeRuntime(), metrics)
    spec = make_spec()
    pool.invoke(spec, {})
    pool.invoke(spec, {})

    summary = metrics.summary("hello")
    assert summary["invocations"] == 2
    assert summary["cold_starts"] == 1
    kinds = [e.kind for e in metrics.events()]
    assert "start" in kinds
    pool.shutdown()


def test_the_start_event_carries_where_the_cold_start_went():
    """So a platform at zero can still answer the question.

    The phase breakdown lives on the running container, which is exactly
    what a scale-to-zero platform does not have when it is doing its job.
    Keeping it on the event means the console can take a cold start apart
    after the container that paid for it has been reaped.
    """
    metrics = Metrics()
    pool = Pool(FakeRuntime(start_delay_s=0.05), metrics)
    spec = make_spec(idle_timeout_s=60)
    pool.invoke(spec, {})
    pool.scale_to_zero(spec.name)

    assert pool.instances(spec.name) == [], "nothing is running to ask"
    started = [e for e in metrics.events() if e.kind == "start"]
    assert len(started) == 1
    phases = started[0].phases
    assert phases is not None, "the start event forgot what the start cost"
    assert set(phases) >= {"create_ms", "start_ms", "port_ms", "ready_ms", "total_ms"}
    assert phases["total_ms"] == pytest.approx(started[0].cold_start_ms)
    # The phases are what the total is made of, give or take the rounding
    # each of them carries into the dict.
    parts = sum(phases[k] for k in ("create_ms", "start_ms", "port_ms", "ready_ms"))
    assert parts == pytest.approx(phases["total_ms"], abs=0.05)
    pool.shutdown()


def test_a_stop_event_carries_no_phases_to_confuse_them_with():
    metrics = Metrics()
    pool = Pool(FakeRuntime(), metrics)
    spec = make_spec()
    pool.invoke(spec, {})
    pool.scale_to_zero(spec.name)

    stopped = [e for e in metrics.events() if e.kind == "stop"]
    assert len(stopped) == 1
    assert stopped[0].phases is None
    assert stopped[0].reason == "manual"
    pool.shutdown()


def test_the_reaper_thread_runs_and_stops_cleanly():
    runtime = FakeRuntime()
    pool = Pool(runtime, Metrics(), PoolConfig(reap_interval_s=0.05))
    pool.start_reaper()
    pool.start_reaper()  # idempotent

    pool.invoke(make_spec(idle_timeout_s=0.05), {})
    deadline = time.time() + 3
    while time.time() < deadline and pool.instances("hello"):
        time.sleep(0.05)

    assert pool.instances("hello") == []
    pool.shutdown()
    assert pool._reaper is not None and not pool._reaper.is_alive()
