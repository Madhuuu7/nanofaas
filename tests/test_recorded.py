"""The recording the published console reads.

The deployed page is this console with a recording where the platform would
be, so the recording is a deployed artefact and gets tested like one. What
these tests cannot do is make the recording that ships: that one is driven
against a real daemon by ``scripts/record_console.py`` in the Pages build, and
a recording assembled from fakes would be a drawing of a platform rather than
one. What they can do is drive the same script end to end against a runtime
that answers like the example functions, which covers the part most likely to
break quietly -- the shape of the bundle, the audit that decides whether it is
publishable, and the page's own reading of it -- in a couple of seconds and
with no Docker anywhere.

The division is deliberate. ``audit`` is the gate on *what the session
showed*; ``N.replayProblems`` in the page is the gate on *what the file
holds*; and the two are checked here against the same bundle, because a
recording that satisfies one and not the other is exactly the kind of thing
that reaches a published page.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest
from fakes import faked_platform

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts" / "check_console.mjs"


def _load_recorder():
    """Import the script by path: ``scripts`` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "record_console", ROOT / "scripts" / "record_console.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rc = _load_recorder()


def _record(tmp_path: Path, **session_kwargs):
    """The real session script, against a runtime with no daemon under it."""
    with faked_platform():
        with rc.Server(tmp_path / "data") as base:
            with httpx.Client(base_url=base, timeout=60.0) as http:
                examples = http.get("/api/examples").json()
                recorder = rc.Recorder(http)
                rc.record_session(recorder, quick=True, **session_kwargs)
    return recorder.steps, examples


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """One session, reused: it is the slowest thing in this file."""
    tmp_path = tmp_path_factory.mktemp("recording")
    steps, examples = _record(tmp_path, duration_s=3.0, rate_per_min=600.0, min_requests=24)
    out = tmp_path / "site" / "recorded"
    rc.write_bundle(
        out,
        steps,
        examples,
        host={"docker": "fake", "docker_api": "n/a", "kernel": "n/a",
              "os": "a fake runtime, no daemon", "python": "3.13", "cpus": 4},
        quick=True,
        elapsed_s=3.0,
    )
    shutil.copy2(ROOT / "web" / "index.html", tmp_path / "site" / "index.html")
    return out


# -- the session ----------------------------------------------------------


def test_the_session_covers_every_claim_the_console_makes(recorded):
    session = json.loads((recorded / "session.json").read_text(encoding="utf-8"))
    kinds = [step["kind"] for step in session["steps"]]
    for kind in ("boot", "deploy", "invoke", "burst", "zero", "fail", "traffic",
                 "advise"):
        assert kind in kinds, f"the session never {kind}s"
    # The order is the argument the page makes: nothing running, then a cold
    # start, then the same call warm. Reorder those three and the page is
    # showing a different platform.
    assert kinds[0] == "boot"
    assert kinds[1] == "deploy"
    first, second = session["steps"][2], session["steps"][3]
    assert first["response"]["body"]["cold"] is True
    assert second["response"]["body"]["cold"] is False
    assert (
        first["response"]["body"]["total_ms"]
        > second["response"]["body"]["total_ms"] * 5
    )


def test_every_step_carries_the_five_snapshots_the_page_polls(recorded):
    session = json.loads((recorded / "session.json").read_text(encoding="utf-8"))
    for step in session["steps"]:
        assert set(step["state"]) == set(rc.STATE), step["label"]
        assert step["state"]["health"]["docker"] is True
        assert step["label"] and step["note"]


def test_the_burst_scaled_out_and_scale_to_zero_brought_it_back(recorded):
    session = json.loads((recorded / "session.json").read_text(encoding="utf-8"))
    steps = {step["kind"]: step for step in session["steps"]}
    burst = steps["burst"]
    assert burst["response"]["body"]["ok"] == 10
    assert len(burst["state"]["instances"]) > 1, "the burst ran one container"
    assert steps["zero"]["response"]["body"]["stopped"] >= 1
    assert steps["zero"]["state"]["instances"] == []


def test_the_recommendation_is_drawn_from_the_recorded_traffic(recorded):
    session = json.loads((recorded / "session.json").read_text(encoding="utf-8"))
    advises = [s for s in session["steps"] if s["kind"] == "advise"]
    assert [s["rate_pct"] for s in advises] == [5, 1]
    for advise in advises:
        body = advise["response"]["body"]
        assert body["observed"]["requests"] >= 20
        assert body["sweep"], "a recommendation with no sweep behind it"
        assert body["explanation"]
    # A stricter target cannot ask for a shorter timeout.
    at_5, at_1 = (a["response"]["body"]["recommended"]["idle_timeout_s"] for a in advises)
    assert at_1 >= at_5


def test_the_manifest_indexes_the_session(recorded):
    session = json.loads((recorded / "session.json").read_text(encoding="utf-8"))
    manifest = json.loads((recorded / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == session["format"] == rc.FORMAT
    assert manifest["recorded_at"] == session["recorded_at"]
    assert [s["label"] for s in manifest["steps"]] == [
        s["label"] for s in session["steps"]
    ]


# -- the audit, which is what refuses to publish a dud --------------------


def test_the_audit_passes_a_complete_session(recorded):
    session = json.loads((recorded / "session.json").read_text(encoding="utf-8"))
    assert rc.audit(session["steps"]) == []


def test_the_audit_refuses_a_session_with_nothing_in_it():
    assert "no boot step was recorded" in rc.audit([])


def test_the_audit_refuses_a_recommendation_made_from_noise(tmp_path):
    """The failure mode the traffic phase exists to prevent.

    A handful of requests will produce a recommendation -- the endpoint is
    happy from ten rows up, and the burst alone clears that -- and it is a
    recommendation about nothing. Shipping it would be the most plausible way
    for this console to say something false, so the audit has to catch it
    rather than the reader.
    """
    steps, _ = _record(tmp_path, duration_s=0.05, rate_per_min=600.0, min_requests=1)
    problems = rc.audit(steps)
    assert any("recommendation about noise" in problem for problem in problems), problems


def test_the_audit_refuses_a_session_where_nothing_was_cold(recorded):
    session = json.loads((recorded / "session.json").read_text(encoding="utf-8"))
    steps = json.loads(json.dumps(session["steps"]))
    for step in steps:
        if step["kind"] == "invoke" and step["response"]:
            step["response"]["body"]["cold"] = False
    problems = rc.audit(steps)
    assert any("cold start" in problem for problem in problems), problems


def test_the_audit_refuses_a_burst_that_did_not_scale_out(recorded):
    session = json.loads((recorded / "session.json").read_text(encoding="utf-8"))
    steps = json.loads(json.dumps(session["steps"]))
    for step in steps:
        step["state"]["instances"] = step["state"]["instances"][:1]
    problems = rc.audit(steps)
    assert any("scaling out" in problem for problem in problems), problems


# -- the page's own reading of it ------------------------------------------


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_console_harness_accepts_the_bundle(recorded):
    """The same check the Pages build runs before it deploys anything."""
    result = subprocess.run(
        ["node", str(HARNESS), "--recorded", str(recorded)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "recording in" in result.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_console_harness_rejects_a_bundle_the_page_cannot_read(recorded, tmp_path):
    """A truncated recording has to fail the build, not the visitor."""
    broken = tmp_path / "broken"
    broken.mkdir()
    session = json.loads((recorded / "session.json").read_text(encoding="utf-8"))
    session["steps"] = session["steps"][:3]
    (broken / "session.json").write_text(json.dumps(session), encoding="utf-8")
    shutil.copy2(recorded / "manifest.json", broken / "manifest.json")
    result = subprocess.run(
        ["node", str(HARNESS), "--recorded", str(broken)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 1
    assert "FAILED" in result.stdout + result.stderr


# -- the recorder's own refusal --------------------------------------------


def test_the_recorder_refuses_to_run_without_a_daemon(monkeypatch, tmp_path):
    """No daemon, no recording -- and a message that says why."""
    import docker

    monkeypatch.setattr(
        docker, "from_env",
        lambda *a, **k: (_ for _ in ()).throw(
            docker.errors.DockerException("Cannot connect to the Docker daemon")
        ),
    )
    monkeypatch.setattr("sys.argv", ["record_console.py", "--out", str(tmp_path)])
    assert rc.main() == 1
    assert not list(tmp_path.glob("*.json"))
