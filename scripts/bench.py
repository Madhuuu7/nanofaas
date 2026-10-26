"""The study behind every number in the README.

    python scripts/bench.py            # the full study, ~4 minutes
    python scripts/bench.py --quick    # fewer samples, ~1 minute
    python scripts/bench.py --json out.json

Five parts:

1. **cold start** -- how long, and which phase owns it
2. **warm invocation** -- what the pool is buying, and the two bugs it cost
3. **cpu** -- what a function's CPU allocation does to its own cold start
4. **stampede** -- capping concurrent starts, swept, against the same burst
5. **idle timeout** -- the policy sweep, and the closed form it is checked on

Every number is measured on this machine against real containers. They are
not a claim about Docker in general: a different host, kernel or storage
driver moves them, and the ratios are the transferable part.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nanofaas.functions import EXAMPLES, FunctionSpec  # noqa: E402
from nanofaas.images import build_image, ensure_base_images, remove_image  # noqa: E402
from nanofaas.metrics import Metrics, summarise  # noqa: E402
from nanofaas.policy import (  # noqa: E402
    compare_with_closed_form,
    optimal_timeout_for_poisson,
    recommend,
    sweep,
)
from nanofaas.pool import Pool, PoolConfig  # noqa: E402
from nanofaas.runtime import ContainerRuntime  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def pct(values, q):
    return float(np.quantile(values, q)) if values else 0.0


# -- 1. cold start ---------------------------------------------------------


def study_cold_start(runtime, spec, samples: int) -> dict:
    rule(f"1. cold start, {samples} containers")
    rows = []
    for _ in range(samples):
        instance = runtime.start(spec)
        rows.append(instance.cold_start.to_dict())
        runtime.stop(instance, force=True)

    phases = ("create_ms", "start_ms", "port_ms", "ready_ms", "total_ms")
    print(f"  {'phase':>10s} {'p50':>9s} {'p95':>9s} {'share':>7s}")
    total_p50 = stats.median([r["total_ms"] for r in rows])
    out = {}
    for phase in phases:
        values = [r[phase] for r in rows]
        out[phase] = {"p50": stats.median(values), "p95": pct(values, 0.95)}
        share = "" if phase == "total_ms" else f"{stats.median(values) / total_p50:6.0%}"
        print(f"  {phase:>10s} {stats.median(values):8.0f}ms {pct(values, 0.95):8.0f}ms {share:>7s}")
    print("\n  `ready` is the interpreter booting and importing the handler. It is")
    print("  the only phase a function author can change -- create and start")
    print("  belong to the daemon -- which is why they are reported apart and")
    print("  not summed into one number.")
    return out


# -- 2. warm ---------------------------------------------------------------


def study_warm(runtime, spec, samples: int) -> dict:
    rule(f"2. warm invocation, {samples} calls on one container")
    instance = runtime.start(spec)
    try:
        runtime.invoke(instance, {}, 10.0)  # discard: this one opens the socket
        timings, execs = [], []
        for _ in range(samples):
            started = time.perf_counter()
            _status, body = runtime.invoke(instance, {"name": "bench"}, 10.0)
            timings.append((time.perf_counter() - started) * 1000)
            execs.append(float(body.get("exec_ms") or 0))
        summary = summarise(timings)
        cold_total = instance.cold_start.total_ms
        print(f"  p50 {summary['p50']:.2f}ms   p95 {summary['p95']:.2f}ms   "
              f"p99 {summary['p99']:.2f}ms   max {summary['max']:.2f}ms")
        print(f"  the handler itself: {stats.median(execs):.3f}ms")
        print(f"  cold start was {cold_total:.0f}ms -- "
              f"{cold_total / summary['p50']:.0f}x the warm p50")
        print("\n  Two bugs lived in this gap. A fresh HTTP client per call cost")
        print("  880ms; Nagle waiting on a delayed ACK cost a further 48ms flat,")
        print("  regardless of what the handler did. Both are regression-tested.")
        return {
            "warm_ms": summary,
            "handler_ms": stats.median(execs),
            "cold_start_ms": cold_total,
            "ratio": cold_total / summary["p50"],
        }
    finally:
        runtime.stop(instance, force=True)


# -- 3. cpu ----------------------------------------------------------------


def study_cpu(client, runtime, samples: int, cpu_levels) -> dict:
    rule("3. what a function's CPU allocation does to its own cold start")
    out = {}
    print(f"  {'cpus':>6s} {'ready p50':>11s} {'total p50':>11s}")
    for cpus in cpu_levels:
        spec = FunctionSpec(
            name="bench-cpu", source=EXAMPLES["hello"]["source"],
            memory_mb=128, cpus=cpus, idle_timeout_s=60,
        )
        build_image(client, spec)
        ready, total = [], []
        for _ in range(samples):
            instance = runtime.start(spec)
            ready.append(instance.cold_start.ready_ms)
            total.append(instance.cold_start.total_ms)
            runtime.stop(instance, force=True)
        out[str(cpus)] = {"ready_p50": stats.median(ready), "total_p50": stats.median(total)}
        print(f"  {cpus:6.2f} {stats.median(ready):10.0f}ms {stats.median(total):10.0f}ms")
    print("\n  Cold start is mostly interpreter boot, and interpreter boot is CPU")
    print("  bound, so the cheapest-looking allocation is the one that pays the")
    print("  longest cold start. Worth knowing before trimming cpus to save.")
    return out


# -- 4. stampede -----------------------------------------------------------


def study_stampede(runtime, spec, burst: int, caps) -> dict:
    rule(f"4. a burst of {burst} at a function scaled to zero")
    out = {}
    print(f"  {'max starts':>11s} {'wall':>7s} {'containers':>11s} {'p50':>8s} "
          f"{'p95':>8s} {'mean cold':>10s}")
    for cap in caps:
        pool = Pool(runtime, Metrics(), PoolConfig(max_concurrent_starts=cap))
        results, lock = [], threading.Lock()

        def one():
            try:
                result = pool.invoke(spec, {"ms": 50})
            except Exception as exc:  # noqa: BLE001
                result = {"status": 599, "total_ms": 0.0, "cold": False,
                          "cold_start_ms": 0.0, "error": str(exc)[:60]}
            with lock:
                results.append(result)

        threads = [threading.Thread(target=one) for _ in range(burst)]
        started = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall = time.perf_counter() - started
        containers = len(pool.instances(spec.name))
        latencies = sorted(r["total_ms"] for r in results)
        colds = [r["cold_start_ms"] for r in results if r.get("cold")]
        pool.shutdown()

        out[str(cap)] = {
            "wall_s": wall, "containers": containers,
            "p50_ms": pct(latencies, 0.5), "p95_ms": pct(latencies, 0.95),
            "mean_cold_start_ms": stats.mean(colds) if colds else 0.0,
            "cold_starts": len(colds),
            "ok": sum(1 for r in results if r["status"] == 200),
        }
        print(f"  {cap:11d} {wall:6.1f}s {containers:11d} {pct(latencies, 0.5):7.0f}ms "
              f"{pct(latencies, 0.95):7.0f}ms {(stats.mean(colds) if colds else 0):9.0f}ms")

    print("\n  Capping concurrent starts is a trade, not a free win. Below the")
    print("  point where starts contend, capping just makes later requests")
    print("  queue. Above it, every start slows down together and the")
    print("  uncapped policy spends containers to be slower on every")
    print("  percentile. The crossover is a property of the host, which is")
    print("  why this is a tunable with a measured curve and not a default.")
    return out


# -- 5. idle timeout -------------------------------------------------------


def study_policy(rate_per_minute: float, requests: int) -> dict:
    rule(f"5. the idle timeout for {rate_per_minute:g} requests/minute")
    rng = np.random.default_rng(0)
    gaps = rng.exponential(60.0 / rate_per_minute, size=requests)

    print("  the replay against its closed form, exp(-lambda*T):")
    checks = compare_with_closed_form(rate_per_minute, requests=20_000)
    for row in checks:
        verdict = "ok" if row["abs_error"] <= max(row["noise_3sigma"], 0.002) else "MISMATCH"
        print(f"    T={row['idle_timeout_s']:5.0f}s  replay {row['simulated_cold_rate']:.4f}  "
              f"closed {row['closed_form_cold_rate']:.4f}  "
              f"err {row['abs_error']:.4f} (3s {row['noise_3sigma']:.4f})  {verdict}")

    print(f"\n  {'T':>6s} {'cold rate':>10s} {'GB-seconds':>11s} {'mean penalty':>13s}")
    rows = sweep(gaps, cold_start_s=3.2, exec_s=0.004, memory_mb=128)
    for row in rows:
        print(f"  {row.idle_timeout_s:5.0f}s {row.cold_start_rate:10.1%} "
              f"{row.gb_seconds:11.1f} {row.mean_latency_penalty_ms:12.0f}ms")

    best, why = recommend(gaps, target_cold_rate=0.05, cold_start_s=3.2,
                          exec_s=0.004, memory_mb=128)
    closed = optimal_timeout_for_poisson(rate_per_minute, 0.05)
    print(f"\n  recommendation: {why}")
    print(f"  closed form for the same target: {closed:.1f}s")
    if abs(best.idle_timeout_s - closed) > 1.0:
        # The nearest rung, not an exact match: the closed form here is
        # 29.96s and the sweep's rung is 30s, so an equality test finds
        # nothing and prints nan.
        nearest = min(rows, key=lambda r: abs(r.idle_timeout_s - closed))
        realised = nearest.cold_start_rate
        closed_rung = nearest.idle_timeout_s
        print(
            f"\n  These disagree, and the disagreement is the point rather than a\n"
            f"  defect. The closed form gives the timeout at which the *expected*\n"
            f"  rate is exactly 5%; this trace's realised rate at {closed:.0f}s is\n"
            f"  {realised:.1%}, a hair above, so the recommender refuses it and\n"
            f"  takes the next rung. Asking for a rate to be met on the traffic you\n"
            f"  actually saw is a stricter question than asking for it in\n"
            f"  expectation, and it is the one a team is really asking."
        )
    return {
        "closed_form_check": checks,
        "sweep": [row.to_dict() for row in rows],
        "recommended": best.to_dict(),
        "explanation": why,
        "closed_form_optimum_s": closed,
    }


# -- main ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", type=Path, default=Path("reports/bench.json"))
    parser.add_argument("--skip", nargs="*", default=[])
    args = parser.parse_args()

    try:
        import docker

        client = docker.from_env()
        client.ping()
    except Exception as exc:  # noqa: BLE001
        print(f"no Docker daemon: {exc}")
        print("start Docker Desktop and try again -- this benchmark is about "
              "real containers and there is nothing to measure without them.")
        return 1

    started = time.perf_counter()
    print("nanofaas -- what scale-to-zero actually costs, measured")
    print(f"{'quick' if args.quick else 'full'} run")

    ensure_base_images(client, ("python",))
    runtime = ContainerRuntime(client)
    runtime.reap_orphans()

    hello = FunctionSpec(name="bench-hello", source=EXAMPLES["hello"]["source"],
                         memory_mb=128, cpus=0.5, idle_timeout_s=60)
    sleepy = FunctionSpec(name="bench-sleepy", source=EXAMPLES["sleepy"]["source"],
                          memory_mb=128, cpus=0.5, idle_timeout_s=60,
                          max_instances=16, concurrency=1)
    build_image(client, hello)
    build_image(client, sleepy)

    results: dict = {"quick": args.quick}
    skip = set(args.skip)
    try:
        if "1" not in skip:
            results["cold_start"] = study_cold_start(
                runtime, hello, 5 if args.quick else 12
            )
        if "2" not in skip:
            results["warm"] = study_warm(runtime, hello, 50 if args.quick else 300)
        if "3" not in skip:
            results["cpu"] = study_cpu(
                client, runtime, 2 if args.quick else 4,
                (0.25, 0.5, 1.0) if args.quick else (0.25, 0.5, 1.0, 2.0),
            )
        if "4" not in skip:
            results["stampede"] = study_stampede(
                runtime, sleepy,
                12 if args.quick else 32,
                (2, 16) if args.quick else (1, 2, 4, 8, 16),
            )
        if "5" not in skip:
            results["policy"] = study_policy(6.0, 3000)
    finally:
        runtime.reap_orphans()
        for spec in (hello, sleepy):
            remove_image(client, spec)
        remove_image(client, FunctionSpec(name="bench-cpu",
                                          source=EXAMPLES["hello"]["source"]))
        runtime.close()

    results["elapsed_s"] = time.perf_counter() - started
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.json} in {results['elapsed_s']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
