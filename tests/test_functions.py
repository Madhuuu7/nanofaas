"""The function spec and the registry.

A spec arrives over HTTP from whoever can reach the control plane, so these
tests are mostly about what the platform refuses. The limits are the only
thing standing between "deploy a function" and "allocate a gigabyte and run
forever", and a limit that is documented but not enforced is decoration.
"""

from __future__ import annotations

import json

import pytest

from nanofaas.functions import (
    EXAMPLES,
    MAX_SOURCE_BYTES,
    FunctionSpec,
    Registry,
    SpecError,
)

GOOD = "def handle(event):\n    return {'ok': True}\n"


def make(**overrides) -> FunctionSpec:
    return FunctionSpec(**{"name": "fn-one", "source": GOOD, **overrides})


# -- names -----------------------------------------------------------------


@pytest.mark.parametrize("name", ["fn-one", "abc", "a-b-c-9", "x" * 32])
def test_acceptable_names(name):
    assert make(name=name).name == name


@pytest.mark.parametrize(
    "name",
    [
        "Hello",          # uppercase: container names and tags disagree about it
        "ab",             # too short
        "x" * 33,         # too long
        "9lives",         # must start with a letter
        "trailing-",      # a trailing hyphen is not a valid tag
        "has space",
        "has/slash",      # would change the meaning of an image reference
        "has:colon",      # same
        "",
    ],
)
def test_rejected_names(name):
    with pytest.raises(SpecError, match="invalid name"):
        make(name=name)


# -- limits ----------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        ("memory_mb", 16), ("memory_mb", 99999),
        ("cpus", 0.0), ("cpus", 64),
        ("timeout_s", 0), ("timeout_s", 600),
        ("concurrency", 0), ("concurrency", 1000),
        ("max_instances", 0), ("max_instances", 500),
        ("idle_timeout_s", -1), ("idle_timeout_s", 99999),
    ],
)
def test_limits_outside_the_ceiling_are_refused(field, value):
    """Every one of these is a denial of service with a polite JSON body if
    the ceiling is advisory."""
    with pytest.raises(SpecError, match=field):
        make(**{field: value})


def test_the_defaults_are_inside_their_own_limits():
    """A default that fails its own validation is a platform that cannot
    deploy anything -- worth asserting rather than assuming."""
    spec = FunctionSpec(name="defaults", source=GOOD)
    spec.validate()
    assert spec.memory_mb == 128
    assert spec.idle_timeout_s == 30.0


# -- source ----------------------------------------------------------------


def test_a_source_without_the_entrypoint_is_caught_at_deploy():
    """Otherwise the container starts, the shim cannot find the handler, and
    it presents as a slow cold start rather than a broken function."""
    with pytest.raises(SpecError, match="never defines"):
        make(source="def something_else(event):\n    return {}\n")


def test_a_custom_entrypoint_is_honoured():
    spec = make(source="def main(event):\n    return {}\n", entrypoint="main")
    assert spec.entrypoint == "main"


def test_an_empty_source_is_refused():
    with pytest.raises(SpecError, match="no source"):
        make(source="   \n  \n")


def test_an_oversized_source_is_refused():
    big = "def handle(event):\n    return {}\n" + "# padding\n" * MAX_SOURCE_BYTES
    with pytest.raises(SpecError, match="functions, not for applications"):
        make(source=big)


@pytest.mark.parametrize("key", ["lower", "has-hyphen", "1LEADING", "has space"])
def test_invalid_environment_names_are_refused(key):
    with pytest.raises(SpecError, match="environment variable"):
        make(env={key: "x"})


def test_unsupported_runtime_is_refused():
    with pytest.raises(SpecError, match="unsupported runtime"):
        make(runtime="brainfuck")


# -- round-tripping --------------------------------------------------------


def test_an_unknown_field_is_refused_rather_than_ignored():
    """A typo'd "timeout" silently falling back to the default is how a
    function gets killed at 10 seconds by a platform its author told to wait
    60."""
    with pytest.raises(SpecError, match="unknown fields: timeout"):
        FunctionSpec.from_dict({"name": "fn-one", "source": GOOD, "timeout": 60})


def test_a_spec_survives_a_round_trip():
    original = make(memory_mb=256, idle_timeout_s=5, env={"LOG_LEVEL": "debug"})
    assert FunctionSpec.from_dict(original.to_dict()) == original


def test_the_digest_covers_what_the_image_is_built_from():
    base = make()
    assert make().digest() == base.digest()
    assert make(source=GOOD.replace("True", "False")).digest() != base.digest()
    assert make(entrypoint="handle").digest() == base.digest()
    # Scaling knobs must *not* change the digest, or every tuning tweak costs
    # an image rebuild for no reason.
    assert make(idle_timeout_s=99).digest() == base.digest()
    assert make(max_instances=9).digest() == base.digest()
    assert make(memory_mb=512).digest() == base.digest()


def test_the_image_tag_is_derived_from_the_name():
    assert make(name="my-fn").image_tag == "nanofaas/my-fn:latest"


# -- the registry ----------------------------------------------------------


def test_the_registry_persists_across_instances(tmp_path):
    path = tmp_path / "functions.json"
    first = Registry(path)
    first.put(make(name="alpha"))
    first.put(make(name="beta", memory_mb=256))

    second = Registry(path)
    assert [s.name for s in second.list()] == ["alpha", "beta"]
    assert second.get("beta").memory_mb == 256


def test_a_redeploy_replaces_rather_than_duplicates(tmp_path):
    registry = Registry(tmp_path / "f.json")
    registry.put(make(name="alpha", memory_mb=128))
    registry.put(make(name="alpha", memory_mb=512))
    assert len(registry) == 1
    assert registry.get("alpha").memory_mb == 512


def test_deleting_reports_whether_anything_was_there(tmp_path):
    registry = Registry(tmp_path / "f.json")
    registry.put(make(name="alpha"))
    assert registry.delete("alpha") is True
    assert registry.delete("alpha") is False
    assert registry.get("alpha") is None


def test_a_corrupt_registry_does_not_stop_the_platform(tmp_path):
    """It would take every working function down with it, and the operator
    would be left with a platform that will not start and no way to ask why."""
    path = tmp_path / "f.json"
    path.write_text("{not json at all", encoding="utf-8")
    registry = Registry(path)
    assert len(registry) == 0
    registry.put(make(name="alpha"))
    assert len(registry) == 1


def test_a_record_that_no_longer_validates_is_skipped_not_fatal(tmp_path):
    """A limit tightened between releases must not make the whole registry
    unreadable -- the other functions are still fine."""
    path = tmp_path / "f.json"
    path.write_text(
        json.dumps(
            {
                "functions": [
                    {"name": "BAD NAME", "source": GOOD},
                    make(name="good-one").to_dict(),
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = Registry(path)
    assert [s.name for s in registry.list()] == ["good-one"]


def test_the_registry_is_written_atomically(tmp_path):
    """Written to a temporary and renamed, so a crash mid-write leaves the
    old file intact rather than a truncated one that loads as zero
    functions."""
    path = tmp_path / "f.json"
    registry = Registry(path)
    registry.put(make(name="alpha"))
    assert json.loads(path.read_text(encoding="utf-8"))["functions"]
    assert not path.with_suffix(".tmp").exists()


def test_the_bundled_examples_are_all_valid():
    """They are offered in the console as one-click deploys, so an invalid
    one is a broken first impression."""
    for name, body in EXAMPLES.items():
        spec = FunctionSpec(name=name, source=body["source"])
        spec.validate()
        assert body["description"]
