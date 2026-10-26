"""Fixtures, and the line between unit tests and the ones that need Docker.

Most of this platform can be tested without a container runtime, and it is
worth the effort to keep it that way: the pool's admission logic -- who waits,
who starts a container, who is refused -- is the part most likely to be wrong
and the part a real Docker test would exercise slowest and least
deterministically. A 3-second cold start times out a test suite long before it
covers the fourth interleaving.

So ``FakeRuntime`` stands in for ``ContainerRuntime`` with the same surface and
a configurable, instant "start". Tests that genuinely need a daemon are marked
``@pytest.mark.docker`` and skip when there is none, which keeps the suite
green on a fresh clone and in CI without Docker.
"""

from __future__ import annotations

import pytest

from fakes import FakeRuntime

from nanofaas.functions import EXAMPLES, FunctionSpec


def docker_available() -> bool:
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:  # noqa: BLE001
        return False


HAS_DOCKER = docker_available()

needs_docker = pytest.mark.skipif(
    not HAS_DOCKER, reason="no Docker daemon; start Docker Desktop to run these"
)


@pytest.fixture
def spec() -> FunctionSpec:
    return FunctionSpec(
        name="hello",
        source=EXAMPLES["hello"]["source"],
        idle_timeout_s=60.0,
        max_instances=4,
        concurrency=1,
    )


@pytest.fixture
def fake_runtime() -> FakeRuntime:
    return FakeRuntime()
