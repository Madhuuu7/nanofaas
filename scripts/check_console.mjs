/**
 * A headless check on the console: no browser, no test framework.
 *
 *   node scripts/check_console.mjs
 *   node scripts/check_console.mjs --health h.json --functions f.json
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
 */

import { readFileSync } from "node:fs";
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
  assert(!/<link\b/.test(html), "no stylesheet links expected");
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

// -- report ----------------------------------------------------------------

const label = live ? "live payloads" : "canned fixtures";
if (failures.length) {
  console.error(`console check FAILED on the ${label}`);
  for (const failure of failures) console.error(`  x ${failure.name}\n      ${failure.message}`);
  console.error(`\n${passed} passed, ${failures.length} failed`);
  process.exit(1);
}
console.log(`console check passed on the ${label}: ${passed} checks`);
