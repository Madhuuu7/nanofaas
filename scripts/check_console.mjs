/**
 * A headless check on the console: no browser, no test framework.
 *
 *   node scripts/check_console.mjs
 *   node scripts/check_console.mjs --health h.json --functions f.json
 *   node scripts/check_console.mjs --recorded _site/recorded
 *
 * **Static.** Every id the script reaches for must exist in the markup. A typo
 * there is silent in a browser -- the element is null, the handler throws
 * inside an async function, and the page stops updating with nothing in the
 * terminal. The Python tests cannot see it and neither can anyone who does not
 * open the page. It also asserts the page loads nothing from the network.
 *
 * **Behavioural.** The page's pure helpers run in a Node vm against canned
 * platform payloads. They are the functions that turn container counts and
 * latency samples into claims, and a console that states a wrong claim
 * confidently is worse than one that renders nothing.
 *
 * **The recording.** With ``--recorded``, the bundle a published copy reads
 * is checked the way the page reads it: the snapshots must render, the
 * buttons must reach the steps they claim to, and the session must actually
 * contain the cold start it was published to show. This is the only check
 * that can fail on the deployed artefact rather than on the source, which is
 * why it runs before a deploy and not after one.
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const html = readFileSync(resolve(root, "web/index.html"), "utf8");

let passed = 0;
const failures = [];
const check = (name, fn) => {
  try { fn(); passed += 1; } catch (err) { failures.push({ name, message: err.message }); }
};
const assert = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const assertEqual = (a, b, m) => {
  if (a !== b) throw new Error((m ? m + ": " : "") + `expected ${JSON.stringify(b)}, got ${JSON.stringify(a)}`);
};
const assertIncludes = (h, n, m) => assert(
  String(h).includes(n),
  (m ? m + ": " : "") + `expected ${JSON.stringify(n)} in ${JSON.stringify(String(h).slice(0, 200))}`
);

const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/);
assert(scriptMatch, "no inline <script> in web/index.html");
const source = scriptMatch[1];
const sandbox = { module: { exports: {} }, console };
vm.createContext(sandbox);
new vm.Script(source, { filename: "index.html#script" }).runInContext(sandbox);
const N = sandbox.module.exports;
assert(N && typeof N.fmtMs === "function", "the helpers did not export");

// -- static ---------------------------------------------------------------

const declaredIds = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]));
const referencedIds = new Set([
  ...[...source.matchAll(/\$\("([^"]+)"\)/g)].map((m) => m[1]),
  ...[...source.matchAll(/getElementById\("([^"]+)"\)/g)].map((m) => m[1]),
]);

check("every id the script reaches for exists in the markup", () => {
  const missing = [...referencedIds].filter((id) => !declaredIds.has(id));
  assertEqual(missing.join(", "), "", "ids referenced but never declared");
  assert(referencedIds.size > 12, "suspiciously few ids referenced");
});

check("the page loads nothing from the network", () => {
  const external = [...html.matchAll(/(?:src|href)="([^"]+)"/g)]
    .map((m) => m[1]).filter((u) => /^(https?:)?\/\//.test(u));
  assertEqual(external.join(", "), "", "external resources found");
  // Every <link> has to be inline. The favicon is one: a page demoed offline
  // that references a file it does not ship leaves a 404 in the console of
  // everyone who opens it, which is a small thing that reads as carelessness.
  const links = [...html.matchAll(/<link\b[^>]*href="([^"]+)"/g)].map((m) => m[1]);
  const fetched = links.filter((url) => !url.startsWith("data:"));
  assertEqual(fetched.join(", "), "", "a <link> points at something off the page");
  // The typefaces are embedded, and the way that silently stops being true
  // is a stylesheet url() pointing back at a font host.
  const remote = [...html.matchAll(/url\(\s*['"]?(https?:|\/\/)[^)]*\)/g)].map((m) => m[0]);
  assertEqual(remote.join(", "), "", "a stylesheet fetches something");
  assert(/@font-face/.test(html) && /src: url\(data:font/.test(html),
    "the console should carry its own typefaces");
});

check("every API path the console calls is one the server serves", () => {
  const paths = [...source.matchAll(/api\("(\/api[^"?]*)/g)].map((m) => m[1]);
  const served = [
    "/api/health", "/api/examples", "/api/functions", "/api/instances",
    "/api/invocations", "/api/metrics", "/api/policy/",
  ];
  const unknown = paths.filter((p) => !served.some((s) => p.startsWith(s)));
  assertEqual(unknown.join(", "), "", "console calls an endpoint the API does not serve");
  assert(paths.includes("/api/health"), "the console never checks health");
});

// -- fixtures --------------------------------------------------------------

const health = (over = {}) => Object.assign({
  status: "ok", docker: true, error: null, functions: 2, uptime_s: 90,
  pool: {
    functions: {
      hello: { instances: 1, starting: 0, in_flight: 0, capacity: 1, idle_timeout_s: 30, max_instances: 4 },
      fib: { instances: 0, starting: 0, in_flight: 0, capacity: 0, idle_timeout_s: 30, max_instances: 4 },
    },
    starting_now: 0, max_concurrent_starts: 2,
  },
}, over);

const functions = () => ([
  {
    name: "hello", memory_mb: 128, idle_timeout_s: 30, max_instances: 4,
    live: { instances: 1 },
    metrics: {
      invocations: 50, cold_start_rate: 0.04,
      warm_ms: { n: 48, p50: 3.8, p95: 6.1 },
      cold_ms: { n: 2, p50: 3200, p95: 3400 },
      total_ms: { n: 50, p50: 4.0, p95: 3200 },
    },
  },
]);

const instances = () => ([{
  name: "nanofaas-hello-1-99891", function: "hello", host_port: 9001,
  cold_start: { total_ms: 3200.5 }, age_s: 42.0, idle_s: 12.0,
  invocations: 9, in_flight: 0, poisoned: false,
}]);

const invocations = () => {
  const out = [];
  for (let i = 0; i < 48; i++) {
    out.push({ function: "hello", at: 1700000000 + i, queued_ms: 0.2, cold_start_ms: 0,
               exec_ms: 0.01, total_ms: 3.8 + (i % 5) * 0.3, cold: false, status: 200 });
  }
  out.push({ function: "hello", at: 1700000100, queued_ms: 0.1, cold_start_ms: 3200,
             exec_ms: 0.01, total_ms: 3204, cold: true, status: 200 });
  out.push({ function: "hello", at: 1700000200, queued_ms: 0.1, cold_start_ms: 0,
             exec_ms: 0, total_ms: 12, cold: false, status: 500 });
  return out;
};

const policyReport = () => ({
  function: "hello",
  observed: { requests: 50, measured_cold_start_s: 3.2, mean_exec_s: 0.0001, current_idle_timeout_s: 30 },
  recommended: { idle_timeout_s: 30, cold_start_rate: 0.046, gb_seconds: 3532.5 },
  explanation: "30s holds cold starts at 4.6% for 3532.5 GB-seconds.",
  sweep: [
    { idle_timeout_s: 0, cold_start_rate: 1.0, gb_seconds: 3.8, mean_latency_penalty_ms: 3200 },
    { idle_timeout_s: 10, cold_start_rate: 0.372, gb_seconds: 2366.2, mean_latency_penalty_ms: 1189 },
    { idle_timeout_s: 30, cold_start_rate: 0.046, gb_seconds: 3532.5, mean_latency_penalty_ms: 147 },
    { idle_timeout_s: 300, cold_start_rate: 0.0, gb_seconds: 3748.1, mean_latency_penalty_ms: 1 },
  ],
});

const arg = (flag) => {
  const i = process.argv.indexOf(flag);
  return i > -1 ? JSON.parse(readFileSync(process.argv[i + 1], "utf8")) : null;
};
const liveHealth = arg("--health");
const live = Boolean(liveHealth);

// -- formatting ------------------------------------------------------------

check("durations read at the scale they are", () => {
  assertEqual(N.fmtMs(3.84), "3.8ms");
  assertEqual(N.fmtMs(48), "48ms");
  assertEqual(N.fmtMs(3200), "3.20s");
  assertEqual(N.fmtMs(null), "--");
  assertEqual(N.fmtPct(0.046), "4.6%");
  assertEqual(N.fmtPct(0.046, 0), "5%");
  assertEqual(N.fmtAge(42), "42s");
  assertEqual(N.fmtAge(300), "5.0m");
});

// -- the platform strip ----------------------------------------------------

check("the platform strip counts containers and in-flight work", () => {
  const tiles = N.platformTiles(liveHealth || health());
  const joined = tiles.map((t) => t.label + "=" + t.value).join("|");
  assertIncludes(joined, "containers running");
  assertIncludes(joined, "requests in flight");
  assertIncludes(joined, "max concurrent starts");
  if (!live) assertIncludes(joined, "containers running=1");
});

check("a function at zero containers is called out, not hidden", () => {
  const note = N.platformNote(health());
  assertIncludes(note, "cold start");
  assertIncludes(note, "1 of 2");
});

check("nothing deployed reads as scale-to-zero working, not as an error", () => {
  const note = N.platformNote(health({ functions: 0, pool: { functions: {}, starting_now: 0, max_concurrent_starts: 2 } }));
  assertIncludes(note, "what scale to zero looks like");
});

check("a missing docker daemon is explained rather than rendered as zero", () => {
  const note = N.platformNote(health({
    status: "degraded", docker: false,
    error: "DockerException: Cannot connect to the Docker daemon.",
    pool: undefined,
  }));
  assertIncludes(note, "control plane is up and the data plane is not");
  assertIncludes(note, "Cannot connect");
});

// -- tables ----------------------------------------------------------------

check("a function row carries live containers against the ceiling", () => {
  const rows = N.functionRows(functions());
  assertEqual(rows.length, 1);
  assertEqual(rows[0].instances, 1);
  assertEqual(rows[0].max_instances, 4);
  assertEqual(rows[0].warmP50, 3.8);
  assertEqual(rows[0].coldRate, 0.04);
});

check("a function that has never run does not claim a zero cold rate", () => {
  // null, not 0: "never measured" and "measured at zero" are different
  // claims and the table must not make the first look like the second.
  const rows = N.functionRows([{ name: "fresh", memory_mb: 128, idle_timeout_s: 30,
                                 max_instances: 4, live: {}, metrics: {} }]);
  assertEqual(rows[0].coldRate, null);
  assertEqual(rows[0].warmP50, null);
  assertEqual(rows[0].instances, 0);
});

check("the instance note says which container stops next and when", () => {
  const note = N.instanceNote(instances(), functions());
  assertIncludes(note, "1 container running");
  assertIncludes(note, "stops in 18s");
});

check("a container past its timeout is described as about to stop", () => {
  const stale = instances();
  stale[0].idle_s = 45;
  assertIncludes(N.instanceNote(stale, functions()), "about to be stopped");
});

check("no containers reads as every function at zero", () => {
  assertIncludes(N.instanceNote([], functions()), "Nothing is running");
});

check("recent invocations are newest first and label their outcome", () => {
  const rows = N.recentRows(invocations(), 5);
  assertEqual(rows.length, 5);
  assertEqual(rows[0].status, 500);
  assertEqual(rows[0].ok, false);
  assert(rows.some((r) => r.cold === false), "warm rows expected");
});

// -- the histogram ---------------------------------------------------------

check("the histogram is log-spaced so both populations are visible", () => {
  const hist = N.histogram(invocations(), 20);
  assertEqual(hist.cold, 1);
  assertEqual(hist.warm, 49);
  assert(hist.bins.length === 20, "wrong bin count");
  assert(hist.max > 0, "nothing was binned");
  // Three orders of magnitude apart: on a linear axis the warm population
  // would collapse into the first bin and the chart would show one bar.
  const coldBin = hist.bins.findIndex((b) => b.cold > 0);
  const warmBin = hist.bins.findIndex((b) => b.warm > 0);
  assert(coldBin > warmBin + 5, "the two populations landed in adjacent bins");
  const counted = hist.bins.reduce((t, b) => t + b.cold + b.warm, 0);
  assertEqual(counted, 50, "an invocation fell outside every bin");
});

check("an empty history draws nothing rather than dividing by zero", () => {
  const hist = N.histogram([], 10);
  assertEqual(hist.bins.length, 0);
  assertEqual(hist.max, 0);
});

check("the latency note states the ratio and what the mixture p95 means", () => {
  const note = N.histNote({
    invocations: 50, cold_start_rate: 0.04,
    warm_ms: { n: 48, p50: 3.8 }, cold_ms: { n: 2, p50: 3200 },
    total_ms: { p95: 3200 },
  });
  assertIncludes(note, "two populations");
  assertIncludes(note, "4.0%");
  assert(/factor of \d+/.test(note), "the note should quantify the gap");
});

check("a warm call that queued is not charged to the warm population", () => {
  // The case that produced a false sentence on the published page: ten
  // concurrent requests at a function scaled to zero, three containers
  // started, seven warm requests waiting ~860ms for one of them. Charging
  // the wait to the warm population read as "warm is as slow as cold".
  const note = N.histNote({
    invocations: 15, cold_start_rate: 0.33,
    warm_ms: { n: 10, p50: 854 },
    warm_service_ms: { n: 10, p50: 0.7 },
    warm_queued: 7,
    cold_ms: { n: 5, p50: 876 },
    total_ms: { p95: 1650 },
  });
  assertIncludes(note, "Warm p50 0.7ms");
  assert(/factor of 125\d/.test(note), `the ratio is against service time: ${note}`);
  assertIncludes(note, "7 warm requests also waited");
  assertIncludes(note, "854ms");
  assertIncludes(note, "about capacity");
});

check("the note falls back to the warm total when nothing queued", () => {
  const note = N.histNote({
    invocations: 12, cold_start_rate: 0.08,
    warm_ms: { n: 11, p50: 4.0 }, warm_service_ms: { n: 11, p50: 3.9 },
    warm_queued: 0, cold_ms: { n: 1, p50: 900 }, total_ms: { p95: 900 },
  });
  assert(!note.includes("waited"), "nothing queued, so say nothing about queueing");
  assertIncludes(note, "Warm p50 3.9ms");
});

check("the latency note is honest when there are no cold starts yet", () => {
  const note = N.histNote({
    invocations: 10, cold_start_rate: 0,
    warm_ms: { n: 10, p50: 3.8, p95: 6 }, cold_ms: { n: 0 }, total_ms: { p95: 6 },
  });
  assertIncludes(note, "none cold");
  assert(!note.includes("factor of"), "no ratio without both populations");
});

check("no invocations yet is said plainly", () => {
  assertIncludes(N.histNote({ invocations: 0 }), "No invocations yet");
});

// -- the recommendation ----------------------------------------------------

check("the policy table marks the recommended row", () => {
  const rows = N.policyRows(policyReport());
  assertEqual(rows.length, 4);
  const marked = rows.filter((r) => r.recommended);
  assertEqual(marked.length, 1);
  assertEqual(marked[0].timeout, 30);
});

check("the policy note reports what it measured, not a default", () => {
  const note = N.policyNote(policyReport());
  assertIncludes(note, "50 invocations");
  assertIncludes(note, "3.20s");
  assertIncludes(note, "currently set to 30s");
});

check("missing data degrades to empty rather than throwing", () => {
  assertEqual(N.policyRows(null).length, 0);
  assertEqual(N.policyRows({}).length, 0);
  assertEqual(N.functionRows(null).length, 0);
  assertEqual(N.instanceRows(null).length, 0);
  assertEqual(N.recentRows(null).length, 0);
  assertEqual(N.policyNote(null), "");
});

// -- container lifetimes ---------------------------------------------------
//
// The chart the console is built around, and the helper most able to draw a
// confident picture of something that did not happen.

const lifetimeState = (overrides) => Object.assign({
  instances: [
    {
      name: "nanofaas-hello-4", function: "hello", age_s: 12, idle_s: 1,
      invocations: 3, in_flight: 0,
      cold_start: { create_ms: 70, start_ms: 260, port_ms: 8, ready_ms: 900, total_ms: 1238 },
    },
  ],
  invocations: [
    { function: "hello", at: 1000, instance: "nanofaas-hello-1", total_ms: 1300, cold: true },
    { function: "hello", at: 1002, instance: "nanofaas-hello-1", total_ms: 4, cold: false },
    { function: "hello", at: 1040, instance: "nanofaas-hello-4", total_ms: 1240, cold: true },
  ],
  metrics: {
    events: [
      { function: "hello", instance: "nanofaas-hello-1", kind: "start", at: 1000, cold_start_ms: 1200,
        phases: { create_ms: 60, start_ms: 240, port_ms: 6, ready_ms: 894, total_ms: 1200 } },
      { function: "hello", instance: "nanofaas-hello-1", kind: "stop", at: 1030, reason: "idle", lifetime_s: 30 },
      { function: "hello", instance: "nanofaas-hello-4", kind: "start", at: 1040, cold_start_ms: 1238,
        phases: { create_ms: 70, start_ms: 260, port_ms: 8, ready_ms: 900, total_ms: 1238 } },
    ],
  },
}, overrides || {});

check("a lifetime is built from the events, the requests and what is running", () => {
  const model = N.lifetimes(lifetimeState());
  assertEqual(model.total, 2);
  const [first, second] = model.lanes;
  assertEqual(first.name, "nanofaas-hello-1");
  assertEqual(first.stop, 1030);
  assertEqual(first.reason, "idle");
  assertEqual(first.calls.length, 2);
  assertEqual(second.running, true);
  assertEqual(second.stop, null, "a running container has not stopped");
});

check("the cold start is drawn before the container could serve", () => {
  // A start event is stamped when the container became able to answer, so
  // the 1.2 seconds it cost happened before that moment. Drawing it after
  // would put the expensive part of a container's life after the point it
  // was already serving requests.
  const model = N.lifetimes(lifetimeState());
  const first = model.lanes[0];
  assertEqual(first.ready, 1000);
  assertEqual(Math.round((first.ready - first.born) * 1000), 1200);
  assert(model.t0 <= first.born, "the window has to contain the births");
});

check("now comes from the recording, never from the reader's clock", () => {
  // A published recording is replayed hours or weeks later. Reading the
  // local clock would draw every container as having died long ago and
  // stretch the axis across the gap.
  const model = N.lifetimes(lifetimeState());
  assertEqual(model.now, 1040);
  assert(model.span > 0, "a span of zero divides by zero in the renderer");
});

check("a container that outlived its own start event is still placed", () => {
  // The event history is bounded. A long-lived container can lose its start
  // event and keep its stop event, which carries the lifetime.
  const model = N.lifetimes({
    instances: [], invocations: [],
    metrics: { events: [
      { instance: "old-1", function: "hello", kind: "stop", at: 2000, reason: "idle", lifetime_s: 120 },
    ] },
  });
  assertEqual(model.lanes.length, 1);
  assertEqual(model.lanes[0].born, 1880);
});

check("only the most recent lanes are drawn, and the note says so", () => {
  const events = [];
  for (let i = 0; i < 30; i += 1) {
    events.push({ instance: "c-" + i, function: "hello", kind: "start", at: 1000 + i, cold_start_ms: 900 });
    events.push({ instance: "c-" + i, function: "hello", kind: "stop", at: 1000 + i + 5, reason: "idle", lifetime_s: 5 });
  }
  const model = N.lifetimes({ instances: [], invocations: [], metrics: { events } }, 18);
  assertEqual(model.lanes.length, 18);
  assertEqual(model.total, 30);
  assertEqual(model.hidden, 12);
  assertIncludes(N.lifetimeNote(model), "18 most recent are drawn");
});

check("the lifetime note reports what the starts cost against what was kept", () => {
  const note = N.lifetimeNote(N.lifetimes(lifetimeState()));
  assertIncludes(note, "2 containers have run here");
  assertIncludes(note, "bought no work at all");
  assertIncludes(note, "stopped by the reaper");
  assert(!/undefined|NaN/.test(note), note);
});

check("an empty platform is described as resting, not as broken", () => {
  const note = N.lifetimeNote(N.lifetimes({ instances: [], invocations: [], metrics: { events: [] } }));
  assertIncludes(note, "resting state");
  assert(!note.includes("0 containers"), "do not open with a zero");
});

// -- where a cold start goes -----------------------------------------------

check("the phases are medians, and their shares account for all of it", () => {
  const phases = N.coldPhases(N.lifetimes(lifetimeState()));
  assertEqual(phases.n, 2);
  const labels = phases.parts.map((p) => p.label).join(",");
  assertEqual(labels, "create,start,port,ready");
  const share = phases.parts.reduce((t, p) => t + p.share, 0);
  assert(Math.abs(share - 1) < 1e-9, `shares sum to ${share}`);
  const ready = phases.parts.filter((p) => p.key === "ready_ms")[0];
  assertEqual(ready.whose, "handler");
  assert(ready.share > 0.6, "ready is most of a cold start and the note says so");
});

check("the phase note names the part a handler can actually change", () => {
  const note = N.phaseNote(N.coldPhases(N.lifetimes(lifetimeState())));
  assertIncludes(note, "Median of 2 cold starts");
  assertIncludes(note, "belongs to the daemon");
  assert(!/undefined|NaN/.test(note), note);
});

check("a platform at zero can still say where its cold starts went", () => {
  // The phase breakdown lives on the start event, not only on the running
  // container, so scaling to zero does not erase the answer.
  const state = lifetimeState({ instances: [] });
  const phases = N.coldPhases(N.lifetimes(state));
  assertEqual(phases.n, 2);
  assertIncludes(N.phaseNote(phases), "Median of 2 cold starts");
});

check("no cold start yet is an instruction, not an empty chart", () => {
  const note = N.phaseNote(N.coldPhases(N.lifetimes({ instances: [], invocations: [], metrics: { events: [] } })));
  assertIncludes(note, "Invoke a function that is at zero");
});

// -- the time axis ---------------------------------------------------------

check("the time axis is marked at readable intervals and ends at now", () => {
  const ticks = N.timeTicks({ t0: 0, now: 40, span: 40 });
  assertEqual(ticks[0].label, "0s");
  assertEqual(ticks[ticks.length - 1].label, "now");
  assert(ticks.length >= 3 && ticks.length <= 8, `${ticks.length} ticks is not readable`);
  // Nothing may be close enough to the right-hand end to print on top of
  // "now", which is the one label that has to stay legible.
  const last = ticks[ticks.length - 2];
  assert(40 - last.at > 2, `a tick at ${last.at}s collides with "now"`);
});

check("a long session is not marked every second", () => {
  const ticks = N.timeTicks({ t0: 0, now: 3000, span: 3000 });
  assert(ticks.length <= 8, `${ticks.length} ticks across 50 minutes`);
  assertEqual(ticks[ticks.length - 1].label, "now");
});

// -- the replay ------------------------------------------------------------
//
// A published copy of this page has no platform behind it and reads a
// recording instead. Everything below is about the one failure that costs
// the most and shows the least: a page that loads, renders, and is wrong,
// because the recording did not hold what it asked for.

check("the replay serves state from a snapshot and actions from a step", () => {
  assertEqual(N.replayRoute("/api/health").state, "health");
  assertEqual(N.replayRoute("/api/invocations?limit=2000").state, "invocations");
  assertEqual(N.replayRoute("/api/metrics", "GET").state, "metrics");
  assertEqual(N.replayRoute("/api/instances").state, "instances");
  assert(N.replayRoute("/api/examples").examples, "the examples come from the bundle");
  // A GET of /api/functions is the function list; a POST to it is a deploy.
  // Confusing the two serves a list of functions as a deploy result.
  assertEqual(N.replayRoute("/api/functions", "GET").state, "functions");
  assertEqual(N.replayRoute("/api/functions", "POST").advance.join(), "deploy");
  assertEqual(
    N.replayRoute("/api/functions/hello/invoke", "POST").advance.join(),
    "invoke,fail",
    "pressing Invoke has to be able to reach a recorded failure"
  );
  // Nothing is deployed at step 0, so the select is empty and the path has a
  // hole in it. Pressing Invoke there still means "show me an invocation".
  assertEqual(N.replayRoute("/api/functions//invoke", "POST").advance.join(), "invoke,fail");
  assertEqual(N.replayRoute("/api/functions/hello/scale-to-zero", "POST").advance.join(), "zero");
  assertEqual(N.replayRoute("/api/policy/hello?target_cold_rate=0.05").advance.join(), "advise");
  assert(N.replayRoute("/api/functions/hello", "DELETE").unsupported, "a delete is not replayable");
});

check("the replay knows every endpoint the page calls", () => {
  // The regression this exists for: someone adds a call to the console, the
  // live page is fine, and the published one answers it with an error nobody
  // sees until a visitor presses the button.
  const called = new Set([...source.matchAll(/api\("(\/api[^"?]*)/g)].map((m) => m[1]));
  const answerable = new Set([
    "/api/health", "/api/examples", "/api/functions", "/api/functions/",
    "/api/instances", "/api/invocations", "/api/metrics", "/api/policy/",
  ]);
  const unknown = [...called].filter((path) => !answerable.has(path));
  assertEqual(unknown.join(", "), "", "the page calls something the replay cannot answer");
});

check("stepping forward wraps, so the buttons never stop answering", () => {
  const steps = [{ kind: "boot" }, { kind: "invoke" }, { kind: "zero" }, { kind: "invoke" }];
  assertEqual(N.nextStepIndex(steps, 0, ["invoke"]).index, 1);
  assertEqual(N.nextStepIndex(steps, 1, ["invoke"]).index, 3);
  const wrapped = N.nextStepIndex(steps, 3, ["invoke"]);
  assertEqual(wrapped.index, 1);
  assertEqual(wrapped.wrapped, true);
  assertEqual(N.nextStepIndex(steps, 0, ["advise"]).index, -1);
  const advises = [{ kind: "advise", rate_pct: 5 }, { kind: "advise", rate_pct: 1 }];
  assertEqual(N.nextStepIndex(advises, -1, ["advise"], (s) => s.rate_pct === 1).index, 1);
});

check("a recording that does not hold what the page reads is refused", () => {
  assertEqual(N.replayProblems(null).length, 1);
  assert(N.replayProblems({ format: 2 }).some((p) => p.includes("format")));
  const thin = {
    format: 1, recorded_at: "2026-11-11T00:00:00Z", host: { docker: "27.0" },
    examples: [{ name: "hello" }],
    steps: [{ i: 0, kind: "boot", label: "l", note: "n", state: {} }],
  };
  const problems = N.replayProblems(thin);
  assert(problems.some((p) => p.includes("no health snapshot")), "missing snapshots must be caught");
  assert(problems.some((p) => p.includes("no deploy step")), "missing kinds must be caught");
  // A recording made with no Docker behind it is a drawing of a platform.
  const faked = JSON.parse(JSON.stringify(thin));
  faked.steps[0].state = {
    health: { docker: false }, functions: [], instances: [], invocations: [],
    metrics: { overall: {} },
  };
  assert(
    N.replayProblems(faked).some((p) => p.includes("no Docker")),
    "a recording of a degraded platform must be caught"
  );
});

const recordedDir = (() => {
  const i = process.argv.indexOf("--recorded");
  return i > -1 ? process.argv[i + 1] : null;
})();

if (recordedDir) {
  const session = JSON.parse(readFileSync(resolve(recordedDir, "session.json"), "utf8"));
  const manifest = JSON.parse(readFileSync(resolve(recordedDir, "manifest.json"), "utf8"));

  check("the bundle is the shape the page reads", () => {
    assertEqual(N.replayProblems(session).join("; "), "");
    assertEqual(manifest.format, 1);
    assertEqual(manifest.recorded_at, session.recorded_at);
    assertEqual(manifest.steps.length, session.steps.length);
  });

  check("the page says where its numbers came from", () => {
    const headline = N.replayHeadline(session);
    assertIncludes(headline, "recording");
    assertIncludes(headline, String(session.host.docker));
    assertIncludes(headline, session.recorded_at.slice(0, 10));
    assertEqual(N.replayChips(session, 3).filter((c) => c.current).length, 1);
  });

  check("every snapshot renders without inventing a number", () => {
    session.steps.forEach((step, i) => {
      const state = step.state;
      N.platformTiles(state.health).forEach((tile) => {
        assert(tile.value !== "undefined" && tile.value !== "NaN",
          `step ${i}: the ${tile.label} tile is ${tile.value}`);
      });
      [N.platformNote(state.health), N.instanceNote(state.instances, state.functions),
       N.histNote(state.metrics.overall)].forEach((note) => {
        assert(note.length > 10, `step ${i}: an empty note`);
        assert(!/undefined|NaN/.test(note), `step ${i}: a note reads "${note}"`);
      });
      N.functionRows(state.functions).forEach((row) => {
        assert(row.name && Number.isFinite(row.invocations),
          `step ${i}: function row ${JSON.stringify(row)}`);
      });
      N.instanceRows(state.instances).forEach((row) => {
        assert(row.name && Number.isFinite(row.age), `step ${i}: container row ${JSON.stringify(row)}`);
      });
      N.recentRows(state.invocations, 12).forEach((row) => {
        assert(Number.isFinite(row.total), `step ${i}: invocation row ${JSON.stringify(row)}`);
      });
      assert(N.histogram(state.invocations, 22).bins.length >= 0);

      // The two charts the page opens on, against every moment of the
      // session rather than against one convenient one.
      const model = N.lifetimes(state, 18);
      assert(model.span > 0, `step ${i}: a lifetime window of zero`);
      model.lanes.forEach((lane) => {
        assert(lane.born <= lane.ready, `step ${i}: ${lane.name} was ready before it was born`);
        assert(lane.stop === null || lane.stop >= lane.born,
          `step ${i}: ${lane.name} stopped before it started`);
        assert(lane.born >= model.t0 - 0.001, `step ${i}: ${lane.name} falls off the chart`);
      });
      [N.lifetimeNote(model), N.phaseNote(N.coldPhases(model))].forEach((note) => {
        assert(note.length > 20 && !/undefined|NaN/.test(note), `step ${i}: "${note}"`);
      });
      N.timeTicks(model).forEach((tick) => {
        assert(tick.label && !/undefined|NaN/.test(tick.label), `step ${i}: tick ${tick.label}`);
      });
    });
  });

  check("the recording can say where its cold starts went", () => {
    // Not from the live containers -- from the start events, which is what
    // lets a snapshot taken at zero still answer the question.
    const atZero = session.steps.filter((step) => !step.state.instances.length);
    assert(atZero.length, "the session never reaches zero containers");
    const answered = atZero.filter((step) => N.coldPhases(N.lifetimes(step.state, 18)).n > 0);
    assert(answered.length,
      "no snapshot at zero containers can break down a cold start, so the " +
      "panel is empty exactly when scale-to-zero is being demonstrated");
  });

  check("pressing the buttons walks the recording and every line reads", () => {
    let cursor = 0;
    const press = (kinds, match) => {
      const found = N.nextStepIndex(session.steps, cursor, kinds, match);
      assert(found.index >= 0, `nothing to reach for ${kinds.join("/")}`);
      cursor = found.index;
      return session.steps[cursor];
    };
    assertIncludes(N.deployLine(press(["deploy"]).response.body), "deployed ");
    // The first invocation in the session has to be a cold one, or the page
    // opens on the one claim the whole project is about and cannot show it.
    assertIncludes(N.invokeLine(press(["invoke", "fail"]).response.body), "COLD");
    assertIncludes(N.invokeLine(press(["invoke", "fail"]).response.body), "warm");
    assertIncludes(N.burstLine(press(["burst"]).response.body), "concurrent invocations");
    assertIncludes(N.zeroLine(press(["zero"]).response.body), "stopped ");
    const advise = press(["advise"], (step) => step.rate_pct === 5);
    const rows = N.policyRows(advise.response.body);
    assert(rows.length > 3, "the sweep is too short to show a trade-off");
    assertEqual(rows.filter((row) => row.recommended).length, 1);
    assertIncludes(N.policyNote(advise.response.body), "invocations of this function");
  });

  check("the recording shows the gap it is published to show", () => {
    const bodies = session.steps
      .filter((step) => (step.kind === "invoke" || step.kind === "fail") && step.response)
      .map((step) => step.response.body)
      .filter((body) => body && Number.isFinite(body.total_ms));
    const cold = bodies.filter((body) => body.cold).map((body) => body.total_ms);
    const warm = bodies.filter((body) => !body.cold).map((body) => body.total_ms);
    assert(cold.length && warm.length, "the session has to contain both a cold and a warm call");
    assert(Math.min(...cold) > Math.max(...warm) * 5,
      `the cold start (${Math.min(...cold)}ms) is not clearly slower than a warm call ` +
      `(${Math.max(...warm)}ms), so the page illustrates nothing`);
  });

  check("the published page is the page in this repository", () => {
    const index = resolve(recordedDir, "..", "index.html");
    if (!existsSync(index)) return;
    assertEqual(readFileSync(index, "utf8"), html,
      "the site's index.html is not web/index.html");
  });
}

// -- report ----------------------------------------------------------------

const label = recordedDir ? `recording in ${recordedDir}` : live ? "live payloads" : "canned fixtures";
if (failures.length) {
  console.error(`console check FAILED on the ${label}`);
  for (const failure of failures) console.error(`  x ${failure.name}\n      ${failure.message}`);
  console.error(`\n${passed} passed, ${failures.length} failed`);
  process.exit(1);
}
console.log(`console check passed on the ${label}: ${passed} checks`);
