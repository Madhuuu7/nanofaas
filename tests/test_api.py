"""The control plane.

Mostly contract, with one behavioural test that matters more than the rest:
the platform must come up and stay useful when Docker is not there. A control
plane that refuses to start because its data plane is down is a control plane
you cannot use to find out why it is down, which is exactly the moment you
need it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nanofaas.api.app import create_app

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts" / "check_console.mjs"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app = create_app(tmp_path_factory.mktemp("nanofaas-api"))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def deployed(client):
    """One real deploy, reused: building an image costs a few seconds."""
    health = client.get("/api/health").json()
    if not health["docker"]:
        pytest.skip("no Docker daemon")
    source = client.get("/api/examples").json()[0]["source"]
    response = client.post(
        "/api/functions",
        json={"name": "api-hello", "source": source, "idle_timeout_s": 60,
              "memory_mb": 96},
    )
    assert response.status_code == 200, response.text
    yield "api-hello"
    client.delete("/api/functions/api-hello")


# -- up without Docker -----------------------------------------------------


def test_the_control_plane_starts_without_docker(tmp_path, monkeypatch):
    """Every endpoint answers and health says what is wrong."""
    import docker

    def refuse(*args, **kwargs):
        raise docker.errors.DockerException("Cannot connect to the Docker daemon")

    monkeypatch.setattr(docker, "from_env", refuse)
    app = create_app(tmp_path / "data")
    with TestClient(app) as offline:
        health = offline.get("/api/health").json()
        assert health["status"] == "degraded"
        assert health["docker"] is False
        assert "Docker daemon" in health["error"]
        assert "nothing can be invoked" in health["error"]

        # Reading still works; acting does not, and says why.
        assert offline.get("/api/functions").status_code == 200
        assert offline.get("/api/examples").status_code == 200
        assert offline.get("/api/instances").json() == []
        assert offline.post(
            "/api/functions",
            json={"name": "x-fn", "source": "def handle(e):\n    return {}\n"},
        ).status_code == 503


# -- validation ------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "Bad Name", "source": "def handle(e):\n    return {}\n"},
        {"name": "ok-fn", "source": "def nope(e):\n    return {}\n"},
        {"name": "ok-fn", "source": "def handle(e):\n    return {}\n", "memory_mb": 99999},
        {"name": "ok-fn", "source": "def handle(e):\n    return {}\n", "timeout_s": 600},
        {"name": "ok-fn", "source": "def handle(e):\n    return {}\n", "max_instances": 500},
        {"name": "ok-fn", "source": ""},
    ],
)
def test_a_bad_function_is_refused_with_a_reason(client, payload):
    response = client.post("/api/functions", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"]


def test_an_unknown_function_is_a_404(client):
    assert client.get("/api/functions/nope").status_code == 404
    assert client.post("/api/functions/nope/invoke", json={}).status_code == 404
    assert client.delete("/api/functions/nope").status_code == 404


# -- the lifecycle ---------------------------------------------------------


def test_deploy_invoke_and_scale_to_zero(client, deployed):
    cold = client.post(f"/api/functions/{deployed}/invoke", json={"name": "api"}).json()
    assert cold["ok"] is True
    assert cold["cold"] is True
    assert cold["cold_start_ms"] > 100
    assert cold["body"]["result"] == {"greeting": "hello api"}

    warm = client.post(f"/api/functions/{deployed}/invoke", json={"name": "api"}).json()
    assert warm["cold"] is False
    assert warm["total_ms"] < cold["total_ms"] / 10
    assert warm["instance"] == cold["instance"]

    assert len(client.get("/api/instances").json()) == 1
    assert client.post(f"/api/functions/{deployed}/scale-to-zero").json()["stopped"] == 1
    assert client.get("/api/instances").json() == []

    again = client.post(f"/api/functions/{deployed}/invoke", json={}).json()
    assert again["cold"] is True


def test_a_redeploy_reuses_the_image_when_nothing_changed(client, deployed):
    source = client.get("/api/examples").json()[0]["source"]
    again = client.post(
        "/api/functions",
        json={"name": deployed, "source": source, "idle_timeout_s": 60,
              "memory_mb": 96},
    ).json()
    assert again["image"]["cached"] is True
    assert again["image"]["build_ms"] < 2000


def test_the_function_listing_carries_live_state_and_metrics(client, deployed):
    client.post(f"/api/functions/{deployed}/invoke", json={})
    rows = client.get("/api/functions").json()
    row = next(r for r in rows if r["name"] == deployed)
    assert row["memory_mb"] == 96
    assert "instances" in row["live"]
    assert row["metrics"]["invocations"] > 0


def test_metrics_split_cold_from_warm(client, deployed):
    client.post(f"/api/functions/{deployed}/invoke", json={})
    overall = client.get("/api/metrics").json()["overall"]
    assert overall["invocations"] > 0
    assert "warm_ms" in overall and "cold_ms" in overall
    assert "mean" not in overall["total_ms"]
    events = client.get("/api/metrics").json()["events"]
    assert any(e["kind"] == "start" for e in events)


def test_invocations_can_be_listed_and_bounded(client, deployed):
    rows = client.get("/api/invocations?limit=5").json()
    assert len(rows) <= 5
    assert all("total_ms" in row for row in rows)
    assert client.get("/api/invocations?limit=0").status_code == 422
    assert client.get("/api/invocations?limit=99999").status_code == 422


# -- the recommendation ----------------------------------------------------


def test_a_recommendation_from_four_requests_is_refused(client, tmp_path_factory):
    """A recommendation from almost no traffic is a recommendation about
    noise, and returning one anyway would be the platform's worst habit."""
    health = client.get("/api/health").json()
    if not health["docker"]:
        pytest.skip("no Docker daemon")
    source = client.get("/api/examples").json()[0]["source"]
    client.post("/api/functions", json={"name": "api-quiet", "source": source,
                                        "memory_mb": 96})
    try:
        response = client.get("/api/policy/api-quiet")
        assert response.status_code == 422
        assert "recommendation about noise" in response.json()["detail"]
    finally:
        client.delete("/api/functions/api-quiet")


def test_a_recommendation_arrives_once_there_is_traffic(client, deployed):
    for _ in range(12):
        client.post(f"/api/functions/{deployed}/invoke", json={})
    report = client.get(f"/api/policy/{deployed}?target_cold_rate=0.05").json()

    assert report["observed"]["requests"] >= 12
    assert report["observed"]["measured_cold_start_s"] > 0
    assert report["recommended"]["idle_timeout_s"] >= 0
    assert report["explanation"]
    # The sweep is monotone in both directions, which is what makes the
    # recommended row a choice rather than an artefact.
    rates = [row["cold_start_rate"] for row in report["sweep"]]
    costs = [row["gb_seconds"] for row in report["sweep"]]
    assert rates == sorted(rates, reverse=True)
    assert costs == sorted(costs)


def test_a_recommendation_for_an_unknown_function_is_a_404(client):
    assert client.get("/api/policy/nope").status_code == 404


# -- deletion --------------------------------------------------------------


def test_deleting_a_function_stops_its_containers(client):
    health = client.get("/api/health").json()
    if not health["docker"]:
        pytest.skip("no Docker daemon")
    source = client.get("/api/examples").json()[0]["source"]
    client.post("/api/functions", json={"name": "api-temp", "source": source,
                                        "memory_mb": 96})
    client.post("/api/functions/api-temp/invoke", json={})
    assert any(i["function"] == "api-temp" for i in client.get("/api/instances").json())

    result = client.delete("/api/functions/api-temp").json()
    assert result["containers_stopped"] == 1
    assert client.get("/api/functions/api-temp").status_code == 404


# -- the console -----------------------------------------------------------


def test_the_console_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_the_openapi_schema_describes_every_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    for path in (
        "/api/health", "/api/examples", "/api/functions",
        "/api/functions/{name}", "/api/functions/{name}/invoke",
        "/api/instances", "/api/metrics", "/api/policy/{name}",
    ):
        assert path in paths, path


def test_the_page_is_self_contained():
    """No CDN, no font host, no analytics: a control plane that phones out
    while you are debugging an outage is not one to ship."""
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    fetched = re.findall(r'(?:src|href)="((?:https?:)?//[^"]*)"', html)
    assert fetched == [], f"the page fetches {fetched}"
    # Every <link> has to be inline too. The favicon is a data: URI, and the
    # `http://www.w3.org/2000/svg` inside it is an XML namespace rather than
    # an address -- no browser resolves it -- which is why this looks at what
    # an attribute points at and not at whether the file contains a URL.
    links = re.findall(r'<link\b[^>]*href="([^"]+)"', html)
    assert links, "the page should ship its own favicon rather than 404 for one"
    assert all(link.startswith("data:") for link in links), links


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_console_harness_passes():
    result = subprocess.run(
        [shutil.which("node"), str(HARNESS)],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
