# nanofaas

A scale-to-zero function platform, in about 2,000 lines. Deploy a handler, get
a container with hard memory, CPU and capability limits; idle containers are
stopped; the next request pays a cold start.

The part worth looking at is not that it runs functions — it is that it
**measures its own cold-start tax and picks the idle timeout from your actual
traffic**, instead of shipping a five-minute default that is a guess about
somebody else's workload.

```python
def handle(event):
    return {"greeting": f"hello {event.get('name', 'world')}"}
```

---

## The numbers

From `python scripts/bench.py` on a 16-core laptop running Docker Desktop on
WSL2. A different host moves all of them; the **ratios** are the transferable
part.

### Where a cold start goes

12 containers, `python:3.12-alpine`, 0.5 CPUs:

| phase | p50 | p95 | share |
|---|---|---|---|
| `create` — daemon writes the config and a writable layer | 75ms | 96ms | 6% |
| `start` — fork, namespace, cgroup | 277ms | 377ms | 21% |
| `port` — read back the published port | 7ms | 15ms | 0% |
| **`ready` — interpreter boots, imports the handler** | **983ms** | 1314ms | **75%** |
| total | **1319ms** | 1704ms | |

Reported as four numbers because they belong to different people. `create` and
`start` are the daemon's and no application tuning moves them. `ready` is the
function author's — it is their imports — and it is three quarters of the bill.
Summing them into one figure hides the only actionable part.

### What the warm pool buys

300 calls on one container:

| | |
|---|---|
| warm p50 | **1.64ms** |
| warm p95 | 3.10ms |
| warm p99 | 4.52ms |
| the handler itself | 0.002ms |
| cold start, same function | 1687ms |
| **ratio** | **1028×** |

Two bugs lived in that gap, and both are now regression-tested:

- A fresh `httpx.Client` per call — re-resolving proxies, opening a new TCP
  connection to the published port every time — cost **880ms** per warm
  invocation. One pooled client: 48ms.
- The remaining 48ms was **flat regardless of what the handler did**, which is
  the signature of Nagle's algorithm waiting on a delayed ACK: the shim writes
  headers and body separately and the kernel held the second write.
  `TCP_NODELAY`: 3.8ms.

880ms → 3.8ms, for a handler that executes in 2 microseconds. The tell for the
second one was the flatness; a cost that ignores the work is never the work.

### CPU allocation is a cold-start decision

Cold start is mostly interpreter boot, and interpreter boot is CPU bound:

| cpus | `ready` p50 | total p50 |
|---|---|---|
| 0.25 | 2746ms | 3238ms |
| 0.50 | 1463ms | 1926ms |
| **1.00** | **561ms** | **960ms** |
| 2.00 | 576ms | 963ms |

**4.9× between 0.25 and 1.0 CPUs, and nothing after that.** The allocation
that looks cheapest is the one that pays the longest cold start, and the
plateau at one core says where to stop paying for more.

### Stampede control has an interior optimum

A burst of 32 simultaneous requests at a function scaled to zero, ceiling of
16 instances:

| max concurrent starts | wall | containers | p50 | p95 | mean cold start |
|---|---|---|---|---|---|
| 1 | 3.6s | 2 | 2683ms | 3457ms | 1778ms |
| 2 | 3.2s | 4 | 1995ms | 2725ms | 1561ms |
| **4** | **2.7s** | 6 | **1597ms** | **2601ms** | 1709ms |
| 8 | 4.2s | 13 | 2340ms | 4042ms | 2202ms |
| 16 (uncapped) | 3.8s | 16 | 3048ms | 3708ms | **3212ms** |

This is the result I expected to go the other way. Capping concurrent starts
is **a trade, not a free win**:

- Too low and requests queue behind a start that could have run in parallel.
- Too high and every start slows down together — mean cold start inflates from
  1709ms to **3212ms**, so the uncapped policy spends 16 containers to be
  slower on every percentile than 4 containers were.

The optimum is interior, at 4 on this machine, and it is a property of the
host rather than of the algorithm. So it is a tunable with a measured curve,
not a default to be smug about. An earlier run with a burst of 12 and a
ceiling of 6 had the uncapped policy *winning* — below the contention point,
capping only adds latency.

### Choosing the idle timeout

This is the only real decision in scale-to-zero. For a given arrival trace the
answer is exactly computable: with one instance and a timeout `T`, a request
is warm if and only if the gap since the previous one is at most `T`. So the
sweep is a **replay of the policy, not a model of it**.

At 6 requests/minute, a 3.2s cold start and 128 MB:

| idle timeout | cold-start rate | GB-seconds | mean latency penalty |
|---|---|---|---|
| 0s | 100.0% | 1.5 | 3200ms |
| 5s | 60.4% | 1478.8 | 1934ms |
| 15s | 22.1% | 2924.6 | 706ms |
| 30s | 5.6% | 3591.2 | 178ms |
| **60s** | **0.4%** | 3795.7 | 13ms |
| 300s | 0.0% | 3836.4 | 1ms |

The replay is checked against the algebra it implements. For Poisson arrivals
the cold-start rate is exactly `exp(-λT)`, and over 20,000 requests the two
agree at every timeout to within three sigma of binomial noise:

| T | replay | `exp(-λT)` | error | 3σ |
|---|---|---|---|---|
| 1s | 0.9096 | 0.9048 | 0.0048 | 0.0062 |
| 10s | 0.3674 | 0.3679 | 0.0005 | 0.0102 |
| 30s | 0.0498 | 0.0498 | 0.0001 | 0.0046 |
| 60s | 0.0029 | 0.0025 | 0.0004 | 0.0010 |

Two independent routes to the same number is the only reason to trust the
replay on a real trace, where there is no formula to check it against.

The recommender asks **"the cheapest timeout that holds cold starts under a
target"** rather than "minimise cost plus latency", because the second needs
an exchange rate between rupees and milliseconds that nobody can state
honestly.

One honest wrinkle it reports rather than hides: for a 5% target the closed
form gives 29.96s, and the recommender picks 60s. The trace's *realised* rate
at the 30s rung is 5.6% — a hair above — so it refuses that rung. Meeting a
rate on the traffic you actually saw is a stricter question than meeting it in
expectation, and it is the one a team means.

---

## Running it

Needs Docker. On Windows, Docker Desktop must be running — it does not start
at login by default.

```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -e ".[dev]"

pytest                            # 142 tests, ~65s (12 need Docker, skipped without it)
pytest -m "not docker"            # 130 tests, ~8s, no daemon needed
python scripts/bench.py --quick   # the study above, ~30s
node scripts/check_console.mjs    # the console, headless
uvicorn nanofaas.api.app:app      # the console at http://127.0.0.1:8000
```

The console deploys a function, invokes it, fires a burst of ten, and shows
containers appearing and being reaped on a 2-second poll — plus a log-spaced
latency histogram with cold and warm stacked separately, and the idle-timeout
recommendation for whatever traffic you have just generated.

Deploying by API instead:

```bash
curl -X POST localhost:8000/api/functions -H 'Content-Type: application/json' -d '{
  "name": "hello",
  "source": "def handle(event):\n    return {\"hi\": event.get(\"name\", \"world\")}\n",
  "memory_mb": 128, "idle_timeout_s": 30
}'
curl -X POST localhost:8000/api/functions/hello/invoke -d '{"name":"you"}'
curl localhost:8000/api/policy/hello          # what timeout your traffic argues for
```

---

## How it works

```
request ──> pool.acquire ──> free slot on a warm container?  ──> invoke
                   │                     no
                   ├──> under the start cap and the instance ceiling?
                   │         yes ──> start a container (cold) ──> invoke
                   │         no  ──> wait on a condition variable
                   └──> deadline ──> 429
```

A container runs a **dependency-free shim**: a stdlib HTTP server that imports
the handler once and serves `POST /invoke`. That single import is why a warm
call is milliseconds, and building a function image is a COPY of two files
onto a pinned base — no package install, no network at build time.

The reaper stops any container idle past its function's `idle_timeout_s`. When
the last one goes the function is at zero.

### What a function container cannot do

Memory and CPU are hard caps, the root filesystem is read-only with a 16 MB
tmpfs at `/tmp`, every capability is dropped, `no-new-privileges` is set, pids
are capped at 128, and the process runs as a non-root user from the image.
Ports are published on `127.0.0.1` explicitly — left to default, Docker binds
`0.0.0.0` and every deployed function is reachable from anything on the
network.

**This is a robustness boundary, not a sandbox.** Containers share the host
kernel; a kernel exploit escapes all of it, and the function can still reach
the network because the platform talks to it over a published port. Running
genuinely untrusted code needs a VM boundary — Firecracker or gVisor. See
`deploy/docker-compose.yml` for why mounting the Docker socket into the
control plane is root-equivalent on the host.

---

## Decisions worth arguing about

**A cold start is four numbers.** They have different owners and different
fixes; one number hides which.

**Latency is never reported as a mean.** The distribution is two populations
three orders of magnitude apart. One cold start in a thousand moves the mean
by about the entire warm median, so a mean that moved would be
indistinguishable from a regression. Percentiles are nearest-rank, not
interpolated, for the same reason: interpolating across the gap invents a
latency no request experienced.

**A cold start is attributed only to the request that caused it.** The ones
that queued behind it waited — that is their `queued_ms` — but counting the
start against all of them overstates how often the platform pays one.

**A timed-out invocation poisons its container.** The shim may still be
running the call, so handing it to the next caller serves them a process that
is already busy. A handler that merely *raises* does not poison anything: that
is a function bug and the container is fine.

**An unknown field in a deploy is refused, not ignored.** A typo'd `timeout`
silently falling back to the default is how a function gets killed at 10
seconds by a platform its author told to wait 60.

**The image digest covers the shim and the Dockerfile, not just the handler.**
This was a real bug: editing the shim left every image "cached" against an
unchanged handler digest, so the fix never reached a container while the
deploy reported success.

**The control plane starts without Docker.** Every read still answers and
`/api/health` says what is wrong. A control plane that refuses to start
because its data plane is down is one you cannot use to find out why.

---

## Testing

142 tests, 130 of which need no Docker.

The pool is where this platform is won or lost, and every interesting case is
a race — two threads deciding simultaneously whether to start a container, a
slot freeing while someone waits, a start failing while others queue. Against
real Docker each takes three seconds and resolves differently every run, so
those 27 tests run against a **fake runtime** with an instant, configurable
start: twenty milliseconds, same answer every time.

Three claims cannot be faked and get real containers: the limits are read back
off the running container rather than trusted to the call that set them, the
published port is checked to be unreachable on a non-loopback interface, and a
handler that cannot import is shown to fail the deploy with its traceback.

`scripts/check_console.mjs` runs the console's pure helpers in a Node vm with
no DOM and asserts every id the script reaches for exists in the markup — a
typo there is silent in a browser and invisible to the Python tests.

---

## What this does not do

- **No VM isolation.** See above. This is the honest ceiling on the security
  story.
- **No egress control.** The platform reaches functions over a published port,
  which needs a bridge, so functions can reach the network. Blocking it
  properly means an internal network and a different transport.
- **No secrets.** `env` is passed through in plain text and stored in the
  registry JSON.
- **The idle-timeout model assumes one instance.** With several, warm-or-cold
  depends on which instance a request lands on. The single-instance answer is
  the right one for the low-traffic functions where scale-to-zero actually
  matters, which is the case it is offered for.
- **Invocation is synchronous**, on Starlette's threadpool. A request queued
  behind a cold start blocks a worker thread. A real platform wants an async
  pool and a request queue.
- **The registry is a JSON file.** Fine for a handful of functions on one
  node; it is not a cluster.
- **Python only in practice.** A Node shim and Dockerfile are wired in but
  untested, and `node` is listed in the runtimes rather than claimed as
  working.
- **No Render deploy**, unlike the other projects in this series: a managed
  container platform does not hand you its Docker daemon, which is the one
  thing this is made of.

---

## Layout

```
src/nanofaas/
  functions.py          the spec, its limits, and the registry
  shims/python_shim.py  the dependency-free process inside each container
  images.py             building a function image, and what the digest covers
  runtime.py            container lifecycle and the cold-start breakdown
  pool.py               warm pool, stampede control, scale to zero
  policy.py             choosing the idle timeout from a trace
  metrics.py            the bounded history, by percentile
  api/app.py            the control plane
web/index.html          the console, self-contained
scripts/bench.py        the study behind every number above
scripts/check_console.mjs
deploy/                 compose file, and what the Docker socket costs
```

## License

MIT.
