"""The idle-timeout recommender, checked against the algebra it implements.

The sweep is a replay of the policy rather than a model of it, so most of
these tests are exactness claims: given this trace and this timeout, *this*
many requests were cold, and no sampling is involved. The one statistical test
compares the replay against ``exp(-lambda*T)`` on Poisson arrivals, which is
the only independent check available -- and the reason the replay can be
trusted on a real trace, where there is no formula to compare against.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nanofaas.policy import (
    compare_with_closed_form,
    gaps_from_timestamps,
    optimal_timeout_for_poisson,
    recommend,
    simulate,
    sweep,
)


# -- the replay, exactly ---------------------------------------------------


def test_a_request_is_warm_exactly_when_the_gap_fits_in_the_timeout():
    """The whole policy in one assertion. Gaps of 5s against a 10s timeout
    are warm; the same gaps against a 2s timeout are not."""
    gaps = [5.0, 5.0, 5.0, 5.0]
    warm = simulate(gaps, 10.0)
    cold = simulate(gaps, 2.0)
    # The first request is always cold: nothing is warm at t=0.
    assert warm.cold_starts == 1
    assert cold.cold_starts == 5
    assert warm.requests == cold.requests == 5


def test_a_gap_exactly_equal_to_the_timeout_is_warm():
    """The boundary has to be decided one way and stated. A container idle
    for exactly its timeout has not yet been reaped, so the request lands on
    it."""
    assert simulate([10.0], 10.0).cold_starts == 1  # just the first request
    assert simulate([10.001], 10.0).cold_starts == 2


def test_idle_billing_is_the_gap_when_warm_and_the_timeout_when_not():
    """Where the money goes. A warm gap bills the gap; an expired one bills
    the full timeout and then stops."""
    # Three 4-second gaps under a 10s timeout: 3 * 4s idle, plus the final
    # container waiting out its full timeout after the last request.
    result = simulate([4.0, 4.0, 4.0], 10.0, exec_s=0.0)
    assert result.idle_seconds == pytest.approx(4 * 3 + 10)
    # The same gaps under a 2s timeout: each container expires after 2s.
    expired = simulate([4.0, 4.0, 4.0], 2.0, exec_s=0.0)
    assert expired.idle_seconds == pytest.approx(2 * 3 + 2)


def test_the_last_container_is_billed_for_its_final_idle_period():
    """Easy to forget and pure cost: after the last request of the day, a
    container still sits there for the whole timeout."""
    assert simulate([], 30.0, exec_s=0.0).idle_seconds == pytest.approx(30.0)


def test_scaling_to_zero_immediately_makes_everything_cold():
    result = simulate([1.0, 1.0, 1.0], 0.0)
    assert result.cold_start_rate == 1.0
    assert result.idle_seconds == 0.0


def test_a_long_enough_timeout_removes_every_cold_start_but_the_first():
    result = simulate([1.0] * 100, 3600.0)
    assert result.cold_starts == 1
    assert result.cold_start_rate == pytest.approx(1 / 101)


def test_cost_rises_and_cold_starts_fall_monotonically_with_the_timeout():
    """Both are monotone in T by construction, and a sweep that is not
    monotone means the replay is wrong."""
    rng = np.random.default_rng(0)
    gaps = rng.exponential(10.0, size=2000)
    rows = sweep(gaps, (0, 1, 5, 10, 30, 60, 300))
    rates = [r.cold_start_rate for r in rows]
    costs = [r.gb_seconds for r in rows]
    assert rates == sorted(rates, reverse=True)
    assert costs == sorted(costs)


def test_gb_seconds_scales_with_the_memory_reservation():
    gaps = [5.0] * 50
    small = simulate(gaps, 10.0, memory_mb=128)
    large = simulate(gaps, 10.0, memory_mb=512)
    assert large.gb_seconds == pytest.approx(small.gb_seconds * 4)


def test_the_latency_penalty_is_charged_only_to_cold_requests():
    result = simulate([1.0, 100.0, 1.0], 10.0, cold_start_s=3.0)
    # Two cold: the first request, and the one after the 100s gap.
    assert result.cold_starts == 2
    assert result.mean_latency_penalty_ms == pytest.approx(2 * 3000 / 4)


# -- against the closed form ----------------------------------------------


@pytest.mark.parametrize("rate_per_minute", [2.0, 6.0, 30.0])
def test_the_replay_matches_exp_minus_lambda_t(rate_per_minute):
    """For Poisson arrivals the cold-start rate is exactly exp(-lambda*T).

    Two independent routes to the same number. If the replay drifts from the
    algebra, one of them is wrong, and this says which inputs to look at.
    """
    rows = compare_with_closed_form(rate_per_minute, requests=20_000)
    for row in rows:
        tolerance = max(row["noise_3sigma"], 0.002)
        assert row["abs_error"] <= tolerance, row


def test_the_recommender_agrees_with_the_closed_form_optimum():
    """At 6 requests a minute and a 5% target, -ln(0.05)/lambda is 30.0s, and
    the recommender picks 30s off a simulated trace."""
    rng = np.random.default_rng(1)
    gaps = rng.exponential(10.0, size=4000)  # 6/min
    best, _why = recommend(gaps, target_cold_rate=0.05)
    closed_form = optimal_timeout_for_poisson(6.0, 0.05)
    assert closed_form == pytest.approx(30.0, abs=0.1)
    assert best.idle_timeout_s == pytest.approx(30.0)


def test_the_closed_form_rejects_an_impossible_target():
    for bad in (0.0, 1.0, -0.2, 2.0):
        with pytest.raises(ValueError):
            optimal_timeout_for_poisson(6.0, bad)


def test_a_silent_function_has_an_infinite_optimum():
    assert optimal_timeout_for_poisson(0.0, 0.05) == math.inf


# -- the recommendation ----------------------------------------------------


def test_the_recommendation_is_the_cheapest_timeout_meeting_the_target():
    rng = np.random.default_rng(2)
    gaps = rng.exponential(10.0, size=3000)
    best, why = recommend(gaps, target_cold_rate=0.05)
    assert best.cold_start_rate <= 0.05
    # Nothing cheaper also meets it.
    for row in sweep(gaps):
        if row.gb_seconds < best.gb_seconds:
            assert row.cold_start_rate > 0.05
    assert "GB-seconds" in why


def test_sparse_traffic_is_told_so_rather_than_given_a_bad_answer():
    """Minutes between requests: no timeout in the sweep makes scale-to-zero
    cheap, and saying so is more useful than returning the least-bad row as
    if it were a recommendation."""
    gaps = [1800.0] * 50  # one request every half hour
    best, why = recommend(gaps, target_cold_rate=0.05)
    assert "gaps between requests are long enough" in why
    assert "stop calling it scale-to-zero" in why
    assert best.cold_start_rate > 0.05


def test_a_short_trace_is_told_about_its_floor_not_blamed_for_sparsity():
    """The console found this one. Fourteen requests fired in a burst cannot
    beat a 7.1% cold-start rate, because the first request is always cold --
    nothing is warm at t=0. That is a floor of 1/requests and has nothing to
    do with the gaps, so blaming sparse traffic sends someone to tune a
    timeout that is already doing everything it can.
    """
    gaps = [0.01] * 13  # a dense burst of 14 requests
    best, why = recommend(gaps, target_cold_rate=0.05)
    assert best.cold_starts == 1
    assert "first request is always cold" in why
    assert "floor is 7.1%" in why
    assert "sparse" not in why
    assert "Collect more traffic" in why


def test_a_tighter_target_never_recommends_a_shorter_timeout():
    rng = np.random.default_rng(3)
    gaps = rng.exponential(8.0, size=3000)
    loose, _ = recommend(gaps, target_cold_rate=0.20)
    tight, _ = recommend(gaps, target_cold_rate=0.01)
    assert tight.idle_timeout_s >= loose.idle_timeout_s
    assert tight.gb_seconds >= loose.gb_seconds


# -- trace handling --------------------------------------------------------


def test_gaps_come_from_sorted_timestamps():
    """Invocation records arrive in whatever order the metrics ring held
    them; negative gaps from an unsorted trace would read as warm."""
    gaps = gaps_from_timestamps([30.0, 10.0, 20.0])
    assert gaps == [10.0, 10.0]
    assert all(g >= 0 for g in gaps)


def test_a_single_request_is_one_cold_start():
    result = simulate([], 30.0)
    assert result.requests == 1
    assert result.cold_starts == 1
    assert result.cold_start_rate == 1.0


def test_results_serialise_for_the_api():
    row = simulate([5.0, 5.0], 10.0).to_dict()
    assert row["idle_timeout_s"] == 10.0
    assert set(row) >= {"cold_start_rate", "gb_seconds", "container_seconds"}
    assert all(isinstance(v, (int, float)) for v in row.values())
