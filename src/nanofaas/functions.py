"""What a function is, and what the platform promises it.

A function here is a handler file plus a set of limits. The limits are not
decoration: they are the contract that makes running someone else's code on
your machine a defensible thing to do, and every one of them is enforced at
the container level rather than asked for politely.

    def handle(event: dict) -> dict:
        return {"sum": event["a"] + event["b"]}

That is the whole interface. The platform loads the handler once per
container and reuses it for every invocation that container serves, which is
the entire reason a warm invocation costs single-digit milliseconds and a
cold one costs a second.

## The limits, and why each is there

``memory_mb``        a runaway allocation kills the container, not the host
``cpus``             one function cannot starve the others
``timeout_s``        an infinite loop is reaped rather than owned forever
``concurrency``      how many invocations one container serves at once
``max_instances``    the ceiling on what a traffic spike can cost
``idle_timeout_s``   how long a warm container waits before scaling to zero

The last one is the interesting parameter and the reason this project exists.
Set it too low and every request pays the cold start; too high and you are
paying for idle containers. ``policy.py`` chooses it from measured traffic
rather than from a guess, which is the thing a scale-to-zero platform is
actually for.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Runtime = Literal["python", "node"]

#: Function names become container names, image tags and URL path segments,
#: so they are restricted to what all three accept without escaping.
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]{1,30}[a-z0-9]$")

#: Ceilings the platform will not exceed whatever a spec asks for. A function
#: definition arrives over HTTP, so these are a trust boundary: without them
#: "memory_mb": 99999 is a denial of service with a polite JSON body.
MAX_MEMORY_MB = 1024
MAX_CPUS = 2.0
MAX_TIMEOUT_S = 60.0
MAX_INSTANCES = 20
MAX_SOURCE_BYTES = 256 * 1024


class SpecError(ValueError):
    """A function definition that the platform will not accept."""


@dataclass(frozen=True, slots=True)
class FunctionSpec:
    name: str
    source: str
    runtime: Runtime = "python"
    #: The entry point the shim calls. Kept configurable because "handle" is
    #: a convention, not a law, and a one-line mismatch is a confusing
    #: failure to debug from inside a container.
    entrypoint: str = "handle"
    memory_mb: int = 128
    cpus: float = 0.5
    timeout_s: float = 10.0
    concurrency: int = 1
    max_instances: int = 4
    idle_timeout_s: float = 30.0
    #: Environment handed to the container. Never secrets in this build --
    #: see the README's limits.
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not NAME_PATTERN.match(self.name):
            raise SpecError(
                f"invalid name {self.name!r}: lowercase letters, digits and "
                f"hyphens, 3-32 characters, starting with a letter"
            )
        if self.runtime not in ("python", "node"):
            raise SpecError(f"unsupported runtime {self.runtime!r}")
        if not self.source.strip():
            raise SpecError("the function has no source")
        if len(self.source.encode("utf-8")) > MAX_SOURCE_BYTES:
            raise SpecError(
                f"source is larger than {MAX_SOURCE_BYTES // 1024} KiB; this is a "
                f"platform for functions, not for applications"
            )
        if not re.search(rf"\b{re.escape(self.entrypoint)}\b", self.source):
            raise SpecError(
                f"the source never defines {self.entrypoint!r}; the shim would "
                f"start, fail to find it, and the container would look like a "
                f"slow cold start rather than a broken function"
            )
        _bounded("memory_mb", self.memory_mb, 32, MAX_MEMORY_MB)
        _bounded("cpus", self.cpus, 0.1, MAX_CPUS)
        _bounded("timeout_s", self.timeout_s, 0.1, MAX_TIMEOUT_S)
        _bounded("concurrency", self.concurrency, 1, 64)
        _bounded("max_instances", self.max_instances, 1, MAX_INSTANCES)
        _bounded("idle_timeout_s", self.idle_timeout_s, 0.0, 3600.0)
        for key in self.env:
            if not re.match(r"^[A-Z_][A-Z0-9_]*$", key):
                raise SpecError(f"invalid environment variable name {key!r}")

    @property
    def image_tag(self) -> str:
        return f"nanofaas/{self.name}:latest"

    @property
    def handler_filename(self) -> str:
        return "handler.py" if self.runtime == "python" else "handler.js"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "FunctionSpec":
        known = {f for f in FunctionSpec.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            # Rejected rather than ignored: a typo'd "timeout" silently
            # falling back to the default is how a function gets killed at
            # 10 seconds by a platform the author told to wait 60.
            raise SpecError(f"unknown fields: {', '.join(sorted(unknown))}")
        return FunctionSpec(**payload)

    def digest(self) -> str:
        """What the image depends on. Changing anything here needs a rebuild;
        changing anything else -- the idle timeout, the instance ceiling --
        does not, which keeps a scaling tweak from costing a rebuild."""
        import hashlib

        material = json.dumps(
            {
                "source": self.source,
                "runtime": self.runtime,
                "entrypoint": self.entrypoint,
            },
            sort_keys=True,
        )
        return hashlib.blake2b(material.encode("utf-8"), digest_size=8).hexdigest()


def _bounded(name: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise SpecError(f"{name} must be between {low} and {high}, got {value}")


class Registry:
    """The set of deployed functions, kept on disk so a restart is not a wipe.

    Deliberately a file of JSON rather than a database: the registry is a
    handful of small records read far more often than written, and the
    operational cost of a database here would exceed everything it bought.
    """

    def __init__(self, path: str | Path = "data/functions.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._specs: dict[str, FunctionSpec] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt registry must not stop the platform from starting;
            # it would take every working function down with it.
            return
        for record in payload.get("functions", []):
            try:
                spec = FunctionSpec.from_dict(record)
            except SpecError:
                continue
            self._specs[spec.name] = spec

    def _save(self) -> None:
        payload = {"functions": [spec.to_dict() for spec in self._specs.values()]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # Atomic replace: a crash mid-write leaves the old registry intact
        # rather than a half-written file that loads as zero functions.
        tmp.replace(self.path)

    def put(self, spec: FunctionSpec) -> FunctionSpec:
        with self._lock:
            self._specs[spec.name] = spec
            self._save()
        return spec

    def get(self, name: str) -> FunctionSpec | None:
        with self._lock:
            return self._specs.get(name)

    def delete(self, name: str) -> bool:
        with self._lock:
            existed = self._specs.pop(name, None) is not None
            if existed:
                self._save()
            return existed

    def list(self) -> list[FunctionSpec]:
        with self._lock:
            return sorted(self._specs.values(), key=lambda s: s.name)

    def __len__(self) -> int:
        with self._lock:
            return len(self._specs)


#: Two functions the platform ships with, so the console has something to run
#: on a fresh install and the benchmark has a known workload.
EXAMPLES: dict[str, dict[str, Any]] = {
    "hello": {
        "description": "the cheapest possible function: cold start is all cost",
        "source": (
            "def handle(event):\n"
            "    name = event.get('name', 'world')\n"
            "    return {'greeting': f'hello {name}'}\n"
        ),
    },
    "fib": {
        "description": "CPU-bound, so execution time is visible next to the cold start",
        "source": (
            "def handle(event):\n"
            "    n = int(event.get('n', 25))\n"
            "    if n > 32:\n"
            "        raise ValueError('n too large for a demo function')\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    def rec(k):\n"
            "        return k if k < 2 else rec(k - 1) + rec(k - 2)\n"
            "    return {'n': n, 'iterative': a, 'recursive': rec(min(n, 28))}\n"
        ),
    },
    "sleepy": {
        "description": "sleeps, so queueing behaviour under concurrency is visible",
        "source": (
            "import time\n\n"
            "def handle(event):\n"
            "    ms = min(int(event.get('ms', 200)), 5000)\n"
            "    time.sleep(ms / 1000)\n"
            "    return {'slept_ms': ms}\n"
        ),
    },
}
