"""Turning a handler into a container image.

The build is deliberately trivial: a base image, two files copied in, a
command. No package install, no network, no build step. That keeps a deploy
under a second after the first one and keeps the image small enough that a
cold start is dominated by container creation rather than by reading a
filesystem.

## Why the layers are ordered this way

The shim changes approximately never and the handler changes every deploy, so
the shim is copied first. Docker caches layers in order, which means a
redeploy of the same function reuses everything up to the handler COPY and
rebuilds one tiny layer.

## The security posture, and its limits

Each image runs as a non-root user created at build time. At *run* time --
see ``runtime.py`` -- the container additionally gets no network, a read-only
root filesystem, all capabilities dropped, no new privileges, a pids limit,
and hard memory and CPU caps.

That is defence in depth against a function that misbehaves. It is **not** a
sandbox against hostile code: containers share the host kernel, and a kernel
exploit escapes all of it. A platform that accepted genuinely untrusted code
would need a VM boundary -- Firecracker or gVisor -- and the README says so
rather than implying that a container is one.
"""

from __future__ import annotations

import io
import logging
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

from .functions import FunctionSpec

log = logging.getLogger("nanofaas.images")

SHIMS = Path(__file__).resolve().parent / "shims"

#: Alpine, for size: a smaller image is less to pull once and less page cache
#: to warm on every start. The versions are pinned because "latest" turns a
#: reproducible cold-start measurement into a moving target.
BASE_IMAGES = {
    "python": "python:3.12-alpine",
    "node": "node:20-alpine",
}

DOCKERFILES = {
    # Layer order is most of the performance story of a redeploy. Every RUN
    # after a COPY re-executes when that COPY's content changes, so ownership
    # is set with COPY --chown rather than a trailing RUN chown. Measured on
    # this machine: changing one line of a handler took ~4.0s with the
    # trailing RUN and ~1.6s without it.
    #
    # It does not go much below that, and the reason is worth knowing before
    # optimising further: a build of `FROM python:3.12-alpine` plus a single
    # COPY costs ~0.5s here with everything cached, so most of what remains
    # is the daemon's own per-build overhead rather than anything this
    # Dockerfile does. The adduser layer accounts for ~0.5s of the rest even
    # when cached, and it stays -- running as root would be a worse trade.
    "python": """
FROM {base}
# The user is created before anything is copied, so nothing in the image ends
# up owned by root and writable by the process that runs as fn.
RUN addgroup -S fn && adduser -S -G fn fn
WORKDIR /app
# The shim is identical across every function and every deploy, so its layer
# is built once and reused; the handler, which changes every deploy, is last.
COPY --chown=fn:fn shim.py /app/shim.py
COPY --chown=fn:fn handler.py /app/handler.py
USER fn
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
EXPOSE 8080
CMD ["python", "/app/shim.py"]
""",
    "node": """
FROM {base}
RUN addgroup -S fn && adduser -S -G fn fn
WORKDIR /app
COPY --chown=fn:fn shim.js /app/shim.js
COPY --chown=fn:fn handler.js /app/handler.js
USER fn
EXPOSE 8080
CMD ["node", "/app/shim.js"]
""",
}


@dataclass(frozen=True)
class BuildResult:
    tag: str
    build_ms: float
    cached: bool
    size_mb: float = 0.0
    logs: str = ""


def _tar_context(spec: FunctionSpec) -> io.BytesIO:
    """Build the build context in memory.

    No temporary directory: the context is three small files, and writing
    them to disk only to have the daemon read them back adds failure modes
    (permissions, cleanup on crash) for nothing.
    """
    shim_name = "shim.py" if spec.runtime == "python" else "shim.js"
    shim_source = (SHIMS / f"{spec.runtime}_shim{Path(shim_name).suffix}").read_bytes()
    dockerfile = DOCKERFILES[spec.runtime].format(base=BASE_IMAGES[spec.runtime])

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in (
            ("Dockerfile", dockerfile.encode("utf-8")),
            (shim_name, shim_source),
            (spec.handler_filename, spec.source.encode("utf-8")),
        ):
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 0  # deterministic: identical source gives an identical context
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    buffer.seek(0)
    return buffer


def build_image(client, spec: FunctionSpec, *, pull: bool = False) -> BuildResult:
    """Build (or reuse) the image for a function."""
    tag = spec.image_tag
    started = time.perf_counter()
    existing = None
    try:
        existing = client.images.get(tag)
    except Exception:  # noqa: BLE001 - docker's NotFound, without importing it
        existing = None

    # A cached image is only reusable if it was built from this exact source.
    # Without the digest label a redeploy of changed code would silently keep
    # serving the old one, which is the worst kind of deploy bug: everything
    # reports success and nothing changed.
    if existing is not None:
        labels = existing.labels or {}
        if labels.get("nanofaas.digest") == spec.digest():
            return BuildResult(
                tag=tag,
                build_ms=(time.perf_counter() - started) * 1000.0,
                cached=True,
                size_mb=round(existing.attrs.get("Size", 0) / 1e6, 1),
            )

    context = _tar_context(spec)
    image, logs = client.images.build(
        fileobj=context,
        custom_context=True,
        tag=tag,
        rm=True,
        pull=pull,
        labels={
            "nanofaas.function": spec.name,
            "nanofaas.digest": spec.digest(),
            "nanofaas.runtime": spec.runtime,
        },
    )
    text = "\n".join(
        str(entry.get("stream", "")).rstrip()
        for entry in logs
        if isinstance(entry, dict) and entry.get("stream")
    )
    return BuildResult(
        tag=tag,
        build_ms=(time.perf_counter() - started) * 1000.0,
        cached=False,
        size_mb=round(image.attrs.get("Size", 0) / 1e6, 1),
        logs=text,
    )


def ensure_base_images(client, runtimes=("python",)) -> dict[str, bool]:
    """Pull the base images up front.

    Worth a dedicated call because pulling 50 MB in the middle of the first
    cold start would land in the measurement and make the first number in
    every benchmark a lie about the platform rather than a fact about it.
    """
    out: dict[str, bool] = {}
    for runtime in runtimes:
        base = BASE_IMAGES[runtime]
        try:
            client.images.get(base)
            out[base] = True
        except Exception:  # noqa: BLE001
            log.info("pulling %s", base)
            client.images.pull(base)
            out[base] = False
    return out


def remove_image(client, spec: FunctionSpec) -> bool:
    try:
        client.images.remove(spec.image_tag, force=True)
        return True
    except Exception:  # noqa: BLE001
        return False
