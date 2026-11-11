"""Record one real session of the platform so the console can be published.

    python scripts/record_console.py                 # -> web/recorded/
    python scripts/record_console.py --site _site    # page + recording, deployable
    python scripts/record_console.py --quick         # a shorter traffic phase

The console is a client of a FastAPI control plane whose data plane is the
Docker daemon. GitHub Pages runs neither, so a deployed page has two honest
options: show nothing, or replay a session that really happened. This script
makes the recording for the second.

It is not a fixture file written by hand. It starts the actual server, drives
it over HTTP exactly as the browser does -- deploy, cold invoke, warm invoke,
a burst of ten at a function scaled to zero, scale to zero, the cold start
again, a function that fails, then a traffic phase with exponential gaps so
the idle-timeout recommendation has a trace to reason about -- and after every
step stores the same five GET responses the page polls. Replaying those
snapshots in order is indistinguishable from watching it live, because they
are what was live.

Two consequences worth stating. The numbers belong to whichever host recorded
it, which the bundle names, and a different kernel or storage driver moves
them. And the recording cannot be reproduced byte for byte, because the cold
starts are real; what is checked instead is that the session demonstrated what
it claims to -- a cold start far slower than a warm call, a burst that ran
more than one container, a scale-to-zero that stopped something, a
recommendation drawn from enough requests to mean anything. A recording that
fails those assertions is not published.
"""

from __future__ import annotations

import argparse
import json
import os
import platform as platform_mod
import random
import shutil
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FORMAT = 1

#: The five GETs the console polls. A snapshot is exactly what the page would
#: have seen, so nothing in the replay needs a parallel rendering path.
STATE = {
    "health": "/api/health",
    "functions": "/api/functions",
    "instances": "/api/instances",
    "invocations": "/api/invocations?limit=2000",
    "metrics": "/api/metrics",
}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Server:
    """The real app, on a real port, in a thread.

    Not Starlette's TestClient: the burst step has to be ten genuinely
    concurrent requests, and the whole point of that step is what the pool
    does while several cold starts contend. A transport that serialised them
    would record a platform nobody runs.
    """

    def __init__(self, data_dir: Path):
        import uvicorn

        from nanofaas.api.app import create_app

        self.port = free_port()
        self.server = uvicorn.Server(
            uvicorn.Config(
                create_app(data_dir),
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.server.started:
                return f"http://127.0.0.1:{self.port}"
            if not self.thread.is_alive():
                raise RuntimeError("the server thread died during startup")
            time.sleep(0.05)
        raise RuntimeError("the server did not come up in time")

    def __exit__(self, *_exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=60)


class Recorder:
    def __init__(self, client: httpx.Client):
        self.client = client
        self.steps: list[dict[str, Any]] = []

    def call(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """One request, recorded the way the page's own ``api()`` sees it."""
        response = self.client.request(method, path, **kwargs)
        try:
            body = response.json()
        except ValueError:
            body = {"detail": response.text}
        return {"status": response.status_code, "body": body}

    def snapshot(self) -> dict[str, Any]:
        return {name: self.client.get(path).json() for name, path in STATE.items()}

    def step(
        self,
        kind: str,
        label: str,
        note: str,
        *,
        target: str | None = None,
        request: Any = None,
        response: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "i": len(self.steps),
            "kind": kind,
            "label": label,
            "note": note,
            "target": target,
            "request": request,
            "response": response,
            "state": self.snapshot(),
        }
        if extra:
            record.update(extra)
        self.steps.append(record)
        overall = record["state"]["metrics"]["overall"]
        print(
            f"  [{record['i']:2d}] {label}"
            f"  (invocations {overall.get('invocations', 0)}, "
            f"cold {overall.get('cold_starts', 0)})"
        )
        return record


# -- the session -----------------------------------------------------------


def deploy(
    rec: Recorder, name: str, example: str, **overrides: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    from nanofaas.functions import EXAMPLES

    request: dict[str, Any] = {
        "name": name,
        "source": EXAMPLES[example]["source"],
        "description": EXAMPLES[example]["description"],
        "memory_mb": 128,
        "cpus": 0.5,
        "concurrency": 1,
        "max_instances": 4,
        "idle_timeout_s": 30.0,
    }
    request.update(overrides)
    return rec.call("POST", "/api/functions", json=request), request


def invoke(rec: Recorder, name: str, event: dict[str, Any]) -> dict[str, Any]:
    return rec.call("POST", f"/api/functions/{name}/invoke", json=event)


def record_session(
    rec: Recorder,
    quick: bool,
    *,
    duration_s: float | None = None,
    rate_per_min: float | None = None,
    min_requests: int | None = None,
) -> None:
    """Drive the whole session, recording a snapshot after every step.

    ``duration_s``, ``rate_per_min`` and ``min_requests`` exist for the
    tests, which drive the same script against a fake runtime in a couple of
    seconds. The published recording uses the defaults, and the audit below
    is what stops a too-short run from being deployed as if it were one.
    """
    rec.step(
        "boot",
        "the platform, with nothing deployed",
        "No function has been invoked, so no container exists. Here that is "
        "the resting state, not an outage.",
    )

    # -- hello: the cold-start tax, twice ------------------------------
    response, request = deploy(rec, "hello", "hello")
    if response["status"] != 200:
        raise SystemExit(f"the deploy failed: {response}")
    image = response["body"]["image"]
    built = (
        "reused from the daemon's cache"
        if image["cached"]
        else f"built in {image['build_ms'] / 1000:.1f}s"
    )
    rec.step(
        "deploy",
        "deploy hello",
        f"An image per function, from a dependency-free shim: "
        f"{image['size_mb']} MB, {built}. Nothing is running yet -- a deploy "
        f"builds, it does not start.",
        target="hello",
        request=request,
        response=response,
    )

    first = invoke(rec, "hello", {"name": "world"})
    rec.step(
        "invoke",
        "invoke hello -- cold",
        "Nothing was running, so this request paid for a container: create, "
        "start, then an interpreter booting and importing the handler. That "
        "is the cold-start tax, and managing it is what the rest of this "
        "platform is for.",
        target="hello",
        request={"name": "world"},
        response=first,
    )

    warm = invoke(rec, "hello", {"name": "again"})
    rec.step(
        "invoke",
        "invoke hello -- warm",
        "The same function on the same container, with no start to pay for. "
        "The ratio between this line and the one above it is the only reason "
        "the idle timeout is an interesting question.",
        target="hello",
        request={"name": "again"},
        response=warm,
    )

    # -- the burst at zero ---------------------------------------------
    rec.call("POST", "/api/functions/hello/scale-to-zero")
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [
            pool.submit(invoke, rec, "hello", {"name": f"burst-{i}"})
            for i in range(10)
        ]
        results = [future.result() for future in futures]
    ok = [r for r in results if r["status"] == 200]
    colds = [r for r in ok if r["body"].get("cold")]
    latencies = sorted(r["body"]["total_ms"] for r in ok)
    burst = {
        "requests": len(results),
        "ok": len(ok),
        "cold_starts": len(colds),
        "p50_ms": latencies[len(latencies) // 2] if latencies else None,
        "p95_ms": (
            latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
            if latencies
            else None
        ),
        "max_ms": latencies[-1] if latencies else None,
        "results": [r["body"] for r in results],
    }
    rec.step(
        "burst",
        "ten concurrent requests at a function scaled to zero",
        f"{len(ok)} succeeded, {len(colds)} of them paying a cold start. Ten "
        f"requests did not start ten containers: concurrent starts are "
        f"capped and the rest waited on the starts already in flight. "
        f"Uncapped, the starts contend and every one of them gets slower.",
        target="hello",
        response={"status": 200, "body": burst},
    )

    zeroed = rec.call("POST", "/api/functions/hello/scale-to-zero")
    rec.step(
        "zero",
        "scale hello to zero",
        f"{zeroed['body']['stopped']} container(s) stopped. The reaper does "
        f"this on its own once a container has been idle past its timeout; "
        f"the button only makes the timing yours.",
        target="hello",
        response=zeroed,
    )

    again = invoke(rec, "hello", {"name": "cold again"})
    rec.step(
        "invoke",
        "invoke hello -- cold again",
        "The tax is not a one-off warm-up. Every return from zero pays it in "
        "full, which is why the idle timeout is a real decision rather than a "
        "default to leave alone.",
        target="hello",
        request={"name": "cold again"},
        response=again,
    )

    # -- a function that fails -----------------------------------------
    response, request = deploy(rec, "fib", "fib")
    rec.step(
        "deploy",
        "deploy fib",
        "A CPU-bound handler, so its execution time is visible next to the "
        "cold start instead of being lost inside it.",
        target="fib",
        request=request,
        response=response,
    )
    good = invoke(rec, "fib", {"n": 25})
    rec.step(
        "invoke",
        "invoke fib",
        "Cold start, queueing and execution are reported apart. A platform "
        "that reports one number cannot tell you which of the three to go "
        "and fix.",
        target="fib",
        request={"n": 25},
        response=good,
    )
    bad = invoke(rec, "fib", {"n": 99})
    rec.step(
        "fail",
        "invoke fib with an argument it refuses",
        "The handler raised. The platform reports the function's own failure, "
        "with its traceback, rather than flattening it into a 500 of its "
        "own: the function failed, the platform did not.",
        target="fib",
        request={"n": 99},
        response=bad,
    )

    # -- traffic, so the recommendation has a trace --------------------
    duration = duration_s if duration_s is not None else (60.0 if quick else 150.0)
    rate = rate_per_min if rate_per_min is not None else (18.0 if quick else 7.5)
    # A floor as well as a clock. The gaps are random, so a short run can
    # come in under the number of requests a recommendation needs to mean
    # anything -- and the audit would then reject the recording, after the
    # build had already spent the time making it.
    floor = min_requests if min_requests is not None else (16 if quick else 24)
    rng = random.Random(7)
    started = time.time()
    sent = 0
    while time.time() - started < duration or sent < floor:
        time.sleep(min(rng.expovariate(rate / 60.0), 25.0))
        invoke(rec, "hello", {"name": f"traffic-{sent}"})
        sent += 1
    elapsed = time.time() - started
    rec.step(
        "traffic",
        f"{sent} requests over {elapsed:.0f}s, with exponential gaps",
        f"About {rate:g} requests a minute, the gaps drawn from an "
        f"exponential distribution rather than a fixed interval: real "
        f"arrivals clump, and a timeout tuned on evenly spaced requests is "
        f"tuned on traffic nobody has.",
        target="hello",
        response={
            "status": 200,
            "body": {
                "requests": sent,
                "duration_s": round(elapsed, 1),
                "rate_per_minute": rate,
            },
        },
    )

    for rate in (5, 1):
        report = rec.call("GET", f"/api/policy/hello?target_cold_rate={rate / 100}")
        if report["status"] != 200:
            raise SystemExit(f"the recommendation failed: {report}")
        best = report["body"]["recommended"]
        rec.step(
            "advise",
            f"recommend an idle timeout for a {rate}% cold-start target",
            f"Swept over this function's own arrival gaps and its own "
            f"measured cold start, not a model of either. "
            f"{best['idle_timeout_s']:.0f}s is the cheapest rung that holds "
            f"the cold-start rate at or under {rate}% on the traffic this "
            f"function actually saw.",
            target="hello",
            request={"target_cold_rate": rate / 100},
            response=report,
            extra={"rate_pct": rate},
        )


# -- the assertions that decide whether this is publishable ----------------


def audit(steps: list[dict[str, Any]]) -> list[str]:
    """What a recording has to show before it is worth publishing."""
    problems: list[str] = []
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for step in steps:
        by_kind.setdefault(step["kind"], []).append(step)

    for kind in ("boot", "deploy", "invoke", "burst", "zero", "fail",
                 "traffic", "advise"):
        if kind not in by_kind:
            problems.append(f"no {kind} step was recorded")
    if problems:
        return problems

    invokes = [s for s in by_kind["invoke"] if s["response"]["status"] == 200]
    cold = [s for s in invokes if s["response"]["body"].get("cold")]
    warm = [s for s in invokes if not s["response"]["body"].get("cold")]
    if not cold:
        problems.append("no invocation paid a cold start, so there is no tax to show")
    if not warm:
        problems.append("no invocation was served warm")
    if cold and warm:
        slowest_warm = max(s["response"]["body"]["total_ms"] for s in warm)
        fastest_cold = min(s["response"]["body"]["total_ms"] for s in cold)
        if fastest_cold < slowest_warm * 5:
            problems.append(
                f"the cold start ({fastest_cold:.0f}ms) is not clearly slower "
                f"than a warm call ({slowest_warm:.0f}ms), so the recording "
                f"illustrates nothing"
            )

    burst = by_kind["burst"][0]["response"]["body"]
    if burst["ok"] < burst["requests"]:
        problems.append(
            f"only {burst['ok']} of {burst['requests']} burst requests succeeded"
        )
    through_zero = steps[: by_kind["zero"][0]["i"] + 1]
    containers = max(len(s["state"]["instances"]) for s in through_zero)
    if containers < 2:
        problems.append(
            f"the burst never ran more than {containers} container, so it does "
            f"not show the pool scaling out"
        )
    if by_kind["zero"][0]["response"]["body"]["stopped"] < 1:
        problems.append("scale-to-zero stopped nothing")

    failed = by_kind["fail"][0]["response"]
    if failed["status"] < 400 and failed["body"].get("ok") is not False:
        problems.append("the failing function did not fail")

    for advise in by_kind["advise"]:
        body = advise["response"]["body"]
        if body["observed"]["requests"] < 20:
            problems.append(
                f"the recommendation is drawn from {body['observed']['requests']} "
                f"requests, which is a recommendation about noise"
            )
        if not body["sweep"]:
            problems.append("the recommendation has no sweep behind it")
    return problems


# -- writing the bundle ----------------------------------------------------


def host_facts() -> dict[str, Any]:
    import docker

    client = docker.from_env()
    try:
        version = client.version()
    finally:
        client.close()
    return {
        "docker": version.get("Version"),
        "docker_api": version.get("ApiVersion"),
        "kernel": version.get("KernelVersion"),
        "os": platform_mod.platform(),
        "python": platform_mod.python_version(),
        "cpus": os.cpu_count(),
    }


def write_bundle(
    out: Path,
    steps: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    *,
    host: dict[str, Any],
    quick: bool,
    elapsed_s: float,
) -> dict[str, Any]:
    bundle = {
        "format": FORMAT,
        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": host,
        "quick": quick,
        "elapsed_s": round(elapsed_s, 1),
        "examples": examples,
        "steps": steps,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "session.json").write_text(
        json.dumps(bundle, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    # A small index, so the page can tell a deployed copy from a live one and
    # print what it is replaying before pulling half a megabyte of snapshots.
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "format": FORMAT,
                "recorded_at": bundle["recorded_at"],
                "host": host,
                "elapsed_s": bundle["elapsed_s"],
                "steps": [
                    {key: step[key] for key in ("i", "kind", "label", "target")}
                    for step in steps
                ],
            },
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "web" / "recorded",
        help="where the recording goes (default: web/recorded)",
    )
    parser.add_argument(
        "--site",
        type=Path,
        default=None,
        help="also assemble a deployable site here: the page plus recorded/",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="a shorter traffic phase -- faster, and fewer requests behind "
        "the recommendation",
    )
    args = parser.parse_args()

    try:
        import docker

        client = docker.from_env()
        client.ping()
        client.close()
    except Exception as exc:  # noqa: BLE001
        print(f"no Docker daemon: {exc}")
        print(
            "This records a real session against real containers. There is "
            "nothing to record without a daemon, and a recording assembled "
            "from fixtures would be a drawing of a platform rather than one."
        )
        return 1

    out = args.site / "recorded" if args.site else args.out
    data_dir = ROOT / "data" / "recording"
    if data_dir.exists():
        shutil.rmtree(data_dir)

    print("nanofaas -- recording one real session")
    started = time.perf_counter()
    with Server(data_dir) as base:
        with httpx.Client(base_url=base, timeout=180.0) as http:
            health = http.get("/api/health").json()
            if not health["docker"]:
                print(f"the platform came up degraded: {health['error']}")
                return 1
            examples = http.get("/api/examples").json()
            rec = Recorder(http)
            try:
                record_session(rec, quick=args.quick)
            finally:
                for name in ("hello", "fib"):
                    http.delete(f"/api/functions/{name}")
    elapsed = time.perf_counter() - started

    problems = audit(rec.steps)
    if problems:
        print("\nthis recording is not publishable:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    write_bundle(
        out,
        rec.steps,
        examples,
        host=host_facts(),
        quick=args.quick,
        elapsed_s=elapsed,
    )
    size_kb = (out / "session.json").stat().st_size / 1024
    print(
        f"\nwrote {out / 'session.json'} -- {len(rec.steps)} steps, "
        f"{size_kb:.0f} KB, recorded in {elapsed:.0f}s"
    )

    if args.site:
        args.site.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "web" / "index.html", args.site / "index.html")
        print(f"assembled {args.site} -- index.html plus recorded/")
    shutil.rmtree(data_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
