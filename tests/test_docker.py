"""The parts that only a real daemon can prove.

Everything in ``test_pool.py`` runs against a fake runtime, which is the right
trade for logic. But three claims cannot be tested that way, and they are the
ones that matter if this is ever pointed at code someone else wrote:

* the resource and capability limits are actually applied to the container,
  not merely passed to a function that ignores them
* the published port is bound to loopback and not to every interface
* a handler that cannot import fails the deploy with its traceback, rather
  than presenting as a slow cold start

These are slow -- a cold start is about three seconds -- so there are few of
them and each does several assertions against one container.
"""

from __future__ import annotations

import socket
import time

import pytest

from nanofaas.functions import EXAMPLES, FunctionSpec
from nanofaas.images import build_image, ensure_base_images, image_digest, remove_image
from nanofaas.metrics import Metrics
from nanofaas.pool import Pool, PoolConfig
from nanofaas.runtime import ColdStartFailed, ContainerRuntime

pytestmark = pytest.mark.docker


@pytest.fixture(scope="module")
def client():
    docker = pytest.importorskip("docker")
    try:
        instance = docker.from_env()
        instance.ping()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no Docker daemon: {exc}")
    ensure_base_images(instance, ("python",))
    return instance


@pytest.fixture(scope="module")
def runtime(client):
    rt = ContainerRuntime(client)
    rt.reap_orphans()
    yield rt
    rt.reap_orphans()
    rt.close()


@pytest.fixture(scope="module")
def built(client):
    spec = FunctionSpec(
        name="itest-hello",
        source=EXAMPLES["hello"]["source"],
        memory_mb=96,
        cpus=0.5,
        idle_timeout_s=60,
    )
    build_image(client, spec)
    yield spec
    remove_image(client, spec)


@pytest.fixture
def instance(runtime, built):
    inst = runtime.start(built)
    yield inst
    runtime.stop(inst, force=True)


# -- a container that works ------------------------------------------------


def test_a_container_starts_serves_and_reports_its_cold_start(runtime, built, instance):
    assert instance.host_port > 0
    assert instance.cold_start.total_ms > 0
    # Every phase should be positive and they should sum to the total.
    breakdown = instance.cold_start.to_dict()
    assert breakdown["create_ms"] > 0
    assert breakdown["start_ms"] > 0
    assert breakdown["ready_ms"] > 0
    assert breakdown["total_ms"] == pytest.approx(
        breakdown["create_ms"] + breakdown["start_ms"]
        + breakdown["port_ms"] + breakdown["ready_ms"],
        rel=1e-6,
    )

    status, body = runtime.invoke(instance, {"name": "docker"}, 10.0)
    assert status == 200
    assert body["result"] == {"greeting": "hello docker"}
    assert body["exec_ms"] >= 0


def test_a_warm_invocation_is_orders_faster_than_the_cold_start(runtime, instance):
    """The whole premise of a warm pool, and the regression test for the two
    latency bugs: a fresh HTTP client per call cost 880ms and Nagle cost a
    further 48ms, both for a handler that runs in microseconds."""
    runtime.invoke(instance, {}, 10.0)  # discard the first, it opens the socket
    timings = []
    for _ in range(10):
        started = time.perf_counter()
        status, _body = runtime.invoke(instance, {}, 10.0)
        timings.append((time.perf_counter() - started) * 1000)
        assert status == 200
    timings.sort()
    median = timings[len(timings) // 2]
    assert median < 40, f"warm invocation median {median:.1f}ms: Nagle is back"
    assert median * 20 < instance.cold_start.total_ms


# -- the limits are real ---------------------------------------------------


def test_the_container_runs_with_the_limits_it_was_given(instance, built):
    attrs = instance.container.attrs
    host_config = attrs["HostConfig"]
    assert host_config["Memory"] == built.memory_mb * 1024 * 1024
    assert host_config["NanoCpus"] == int(built.cpus * 1e9)
    assert host_config["PidsLimit"] == 128
    assert host_config["ReadonlyRootfs"] is True
    assert host_config["CapDrop"] == ["ALL"]
    assert "no-new-privileges:true" in (host_config["SecurityOpt"] or [])
    assert "/tmp" in (host_config.get("Tmpfs") or {})


def test_the_function_does_not_run_as_root(runtime, instance):
    """A non-root user is set in the image, so nothing in the container is
    owned by the process that runs there."""
    result = instance.container.exec_run("id -u")
    assert result.exit_code == 0
    assert result.output.decode().strip() != "0"


def test_the_root_filesystem_is_read_only_but_tmp_is_writable(instance):
    denied = instance.container.exec_run("sh -c 'echo x > /app/escape'")
    assert denied.exit_code != 0
    allowed = instance.container.exec_run("sh -c 'echo x > /tmp/fine && cat /tmp/fine'")
    assert allowed.exit_code == 0


def test_the_port_is_published_on_loopback_only(instance):
    """Left to default, Docker binds 0.0.0.0 and every deployed function is
    reachable from anything on the network."""
    ports = instance.container.attrs["NetworkSettings"]["Ports"]["8080/tcp"]
    assert [binding["HostIp"] for binding in ports] == ["127.0.0.1"]

    # And it really is unreachable on a non-loopback address.
    host_ip = socket.gethostbyname(socket.gethostname())
    if host_ip.startswith("127."):
        pytest.skip("no non-loopback address to test against")
    probe = socket.socket()
    probe.settimeout(1.5)
    with pytest.raises(OSError):
        probe.connect((host_ip, instance.host_port))
    probe.close()


# -- failures --------------------------------------------------------------


def test_a_handler_that_cannot_import_fails_the_start_with_its_traceback(
    client, runtime
):
    """Not a slow cold start, not a container restart loop: the reason, at
    deploy time, with the traceback that explains it."""
    spec = FunctionSpec(
        name="itest-broken",
        source="import nonexistent_module_xyz\n\ndef handle(event):\n    return {}\n",
        memory_mb=96,
    )
    build_image(client, spec)
    try:
        with pytest.raises(ColdStartFailed) as excinfo:
            runtime.start(spec)
        message = str(excinfo.value)
        assert "failed to import" in message
        assert "nonexistent_module_xyz" in message
    finally:
        remove_image(client, spec)


def test_a_handler_that_raises_returns_500_with_its_traceback(client, runtime):
    spec = FunctionSpec(
        name="itest-raiser",
        source="def handle(event):\n    raise ValueError('deliberate')\n",
        memory_mb=96,
    )
    build_image(client, spec)
    instance = runtime.start(spec)
    try:
        status, body = runtime.invoke(instance, {}, 10.0)
        assert status == 500
        assert "deliberate" in body["detail"]
    finally:
        runtime.stop(instance, force=True)
        remove_image(client, spec)


def test_a_handler_returning_something_unserialisable_says_so(client, runtime):
    spec = FunctionSpec(
        name="itest-unjson",
        source="def handle(event):\n    return {'f': lambda x: x}\n",
        memory_mb=96,
    )
    build_image(client, spec)
    instance = runtime.start(spec)
    try:
        status, body = runtime.invoke(instance, {}, 10.0)
        assert status == 500
        assert "not JSON" in body["error"]
    finally:
        runtime.stop(instance, force=True)
        remove_image(client, spec)


# -- images ----------------------------------------------------------------


def test_the_image_digest_covers_the_shim_as_well_as_the_handler(client):
    """The bug this guards: editing the shim left every image "cached"
    against an unchanged handler digest, so the fix never reached a container
    while the deploy reported success."""
    spec = FunctionSpec(name="itest-digest", source=EXAMPLES["hello"]["source"])
    try:
        first = build_image(client, spec)
        assert first.cached is False
        assert build_image(client, spec).cached is True

        import nanofaas.images as images

        original = images.DOCKERFILES["python"]
        images.DOCKERFILES["python"] = original + "\n# a change to the build\n"
        try:
            assert image_digest(spec) != first.tag  # sanity: digests are not tags
            assert build_image(client, spec).cached is False
        finally:
            images.DOCKERFILES["python"] = original
    finally:
        remove_image(client, spec)


def test_orphaned_containers_are_reaped(client, runtime, built):
    """A crash leaves running containers with no owner, holding memory and
    ports; a few development crashes would quietly exhaust the machine."""
    instance = runtime.start(built)
    assert instance.container.id
    reaped = runtime.reap_orphans()
    assert reaped >= 1
    remaining = client.containers.list(
        all=True, filters={"label": "nanofaas.managed=true"}
    )
    assert remaining == []


# -- end to end ------------------------------------------------------------


def test_the_pool_scales_to_zero_and_back_on_real_containers(runtime, built):
    """The whole lifecycle against a real daemon: cold, warm, reaped, cold
    again."""
    pool = Pool(runtime, Metrics(), PoolConfig(max_concurrent_starts=2))
    spec = FunctionSpec(**{**built.to_dict(), "idle_timeout_s": 0.5})
    try:
        first = pool.invoke(spec, {"name": "a"})
        assert first["cold"] is True
        assert first["status"] == 200

        second = pool.invoke(spec, {"name": "b"})
        assert second["cold"] is False
        assert second["total_ms"] < first["total_ms"] / 10

        time.sleep(0.7)
        assert pool.reap_idle() == 1
        assert pool.instances(spec.name) == []

        third = pool.invoke(spec, {"name": "c"})
        assert third["cold"] is True
    finally:
        pool.shutdown()
