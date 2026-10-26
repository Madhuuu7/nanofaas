"""The control plane: deploy a function, invoke it, watch it scale to zero.

Two decisions worth stating.

**Invocation is synchronous and runs in the threadpool.** A warm call is about
four milliseconds and a cold one about three seconds, and the handler is
declared ``def`` rather than ``async def`` so Starlette runs it off the event
loop. The pool's own waiting is done on a condition variable, so a request
queued behind a cold start blocks a worker thread and nothing else. A real
platform would want an async pool and a request queue; at this scale the
threadpool is the whole fix, and the README says which one this is.

**The platform does not start Docker for you.** If the daemon is absent, every
endpoint still answers and ``/api/health`` says what is wrong. A control plane
that refuses to start because its data plane is down is a control plane you
cannot use to find out why.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..functions import EXAMPLES, FunctionSpec, Registry, SpecError
from ..images import build_image, ensure_base_images, remove_image
from ..metrics import Metrics
from ..policy import gaps_from_timestamps, recommend, sweep
from ..pool import CapacityTimeout, FunctionFailed, Pool, PoolConfig
from ..runtime import ContainerRuntime

log = logging.getLogger("nanofaas.api")


def _find_root() -> Path:
    override = os.environ.get("NANOFAAS_ROOT")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "web" / "index.html").is_file():
            return parent
    return here.parents[3]


ROOT = _find_root()
WEB = ROOT / "web"


class DeployRequest(BaseModel):
    name: str
    source: str
    runtime: str = "python"
    entrypoint: str = "handle"
    memory_mb: int = Field(default=128, ge=32, le=1024)
    cpus: float = Field(default=0.5, gt=0, le=2.0)
    timeout_s: float = Field(default=10.0, gt=0, le=60.0)
    concurrency: int = Field(default=1, ge=1, le=64)
    max_instances: int = Field(default=4, ge=1, le=20)
    idle_timeout_s: float = Field(default=30.0, ge=0, le=3600)
    description: str = ""


class Platform:
    """Everything with a lifetime longer than one request."""

    def __init__(self, data_dir: Path | str = "data"):
        self.data_dir = Path(data_dir)
        self.registry = Registry(self.data_dir / "functions.json")
        self.metrics = Metrics()
        self.docker = None
        self.runtime: ContainerRuntime | None = None
        self.pool: Pool | None = None
        self.error: str | None = None
        self.started_at = time.time()

    @property
    def ready(self) -> bool:
        return self.pool is not None

    def connect(self) -> None:
        try:
            import docker

            self.docker = docker.from_env()
            self.docker.ping()
            self.runtime = ContainerRuntime(self.docker)
            # Containers from a previous run are orphans: they hold memory
            # and ports and nothing will ever route to them again.
            reaped = self.runtime.reap_orphans()
            if reaped:
                log.info("reaped %d orphaned containers", reaped)
            self.pool = Pool(
                self.runtime,
                self.metrics,
                PoolConfig(
                    max_concurrent_starts=int(
                        os.environ.get("NANOFAAS_MAX_CONCURRENT_STARTS", "2")
                    )
                ),
            )
            self.pool.start_reaper()
            ensure_base_images(self.docker, ("python",))
            self.error = None
        except Exception as exc:  # noqa: BLE001
            self.error = (
                f"{type(exc).__name__}: {exc}. The control plane is up but "
                f"nothing can be invoked until the Docker daemon is reachable."
            )

    def shutdown(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()
        if self.runtime is not None:
            self.runtime.close()

    def require_pool(self) -> Pool:
        if self.pool is None:
            raise HTTPException(status_code=503, detail=self.error or "no docker")
        return self.pool

    def require_function(self, name: str) -> FunctionSpec:
        spec = self.registry.get(name)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"no function {name!r}")
        return spec


def create_app(data_dir: Path | str = "data") -> FastAPI:
    platform = Platform(data_dir)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        platform.connect()
        yield
        platform.shutdown()

    app = FastAPI(
        title="nanofaas",
        version="0.1.0",
        summary="A scale-to-zero function platform that measures its own "
        "cold-start tax and picks the idle timeout from real traffic.",
        lifespan=lifespan,
    )

    # -- platform ------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        body: dict[str, Any] = {
            "status": "ok" if platform.ready else "degraded",
            "docker": platform.ready,
            "error": platform.error,
            "functions": len(platform.registry),
            "uptime_s": round(time.time() - platform.started_at, 1),
        }
        if platform.pool is not None:
            body["pool"] = platform.pool.status()
        return body

    @app.get("/api/examples")
    def examples() -> list[dict[str, Any]]:
        return [
            {"name": name, "description": body["description"], "source": body["source"]}
            for name, body in EXAMPLES.items()
        ]

    # -- functions -----------------------------------------------------

    @app.get("/api/functions")
    def list_functions() -> list[dict[str, Any]]:
        status = platform.pool.status()["functions"] if platform.pool else {}
        out = []
        for spec in platform.registry.list():
            record = spec.to_dict()
            record["live"] = status.get(spec.name, {"instances": 0})
            record["metrics"] = platform.metrics.summary(spec.name)
            out.append(record)
        return out

    @app.post("/api/functions")
    def deploy(request: DeployRequest) -> dict[str, Any]:
        try:
            spec = FunctionSpec(**request.model_dump())
        except SpecError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if platform.docker is None:
            raise HTTPException(status_code=503, detail=platform.error or "no docker")

        started = time.perf_counter()
        try:
            result = build_image(platform.docker, spec)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=422, detail=f"the image did not build: {exc}"
            ) from exc
        platform.registry.put(spec)
        return {
            "function": spec.to_dict(),
            "image": {
                "tag": result.tag,
                "cached": result.cached,
                "build_ms": round(result.build_ms, 1),
                "size_mb": result.size_mb,
            },
            "deploy_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    @app.get("/api/functions/{name}")
    def get_function(name: str) -> dict[str, Any]:
        spec = platform.require_function(name)
        record = spec.to_dict()
        record["instances"] = platform.pool.instances(name) if platform.pool else []
        record["metrics"] = platform.metrics.summary(name)
        return record

    @app.delete("/api/functions/{name}")
    def delete_function(name: str) -> dict[str, Any]:
        spec = platform.require_function(name)
        stopped = platform.pool.scale_to_zero(name) if platform.pool else 0
        if platform.docker is not None:
            remove_image(platform.docker, spec)
        platform.registry.delete(name)
        return {"deleted": name, "containers_stopped": stopped}

    @app.post("/api/functions/{name}/invoke")
    def invoke(name: str, event: dict[str, Any] | None = None) -> dict[str, Any]:
        spec = platform.require_function(name)
        pool = platform.require_pool()
        try:
            result = pool.invoke(spec, event or {})
        except CapacityTimeout as exc:
            # 429, not 500: the platform is working exactly as configured and
            # the caller should back off rather than retry hard.
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except FunctionFailed as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if result["status"] >= 500:
            # The function's own failure is surfaced with its traceback rather
            # than flattened into a platform error.
            return {**result, "ok": False}
        return {**result, "ok": True}

    @app.post("/api/functions/{name}/scale-to-zero")
    def scale_to_zero(name: str) -> dict[str, Any]:
        platform.require_function(name)
        pool = platform.require_pool()
        return {"stopped": pool.scale_to_zero(name)}

    # -- observation ---------------------------------------------------

    @app.get("/api/instances")
    def instances(function: str | None = None) -> list[dict[str, Any]]:
        return platform.pool.instances(function) if platform.pool else []

    @app.get("/api/metrics")
    def metrics(function: str | None = None) -> dict[str, Any]:
        return {
            "overall": platform.metrics.summary(function),
            "by_function": platform.metrics.by_function(),
            "events": [
                {
                    "function": e.function,
                    "instance": e.instance,
                    "kind": e.kind,
                    "at": e.at,
                    "reason": e.reason,
                    "cold_start_ms": round(e.cold_start_ms, 1),
                    "lifetime_s": e.lifetime_s,
                }
                for e in platform.metrics.events(limit=120)
            ],
        }

    @app.get("/api/invocations")
    def invocations(
        function: str | None = None, limit: int = Query(default=200, ge=1, le=5000)
    ) -> list[dict[str, Any]]:
        return [row.to_dict() for row in platform.metrics.invocations(function, limit)]

    @app.delete("/api/metrics")
    def reset_metrics() -> dict[str, str]:
        platform.metrics.reset()
        return {"reset": "ok"}

    # -- the idle-timeout recommendation -------------------------------

    @app.get("/api/policy/{name}")
    def policy(
        name: str,
        target_cold_rate: float = Query(default=0.05, gt=0, lt=1),
    ) -> dict[str, Any]:
        """What idle timeout this function's own traffic argues for."""
        spec = platform.require_function(name)
        rows = platform.metrics.invocations(name)
        if len(rows) < 10:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"only {len(rows)} invocations recorded for {name!r}; a "
                    f"recommendation from that is a recommendation about noise. "
                    f"Drive some traffic at it first."
                ),
            )
        gaps = gaps_from_timestamps([row.at for row in rows])
        cold = [row.cold_start_ms for row in rows if row.cold]
        cold_start_s = (sum(cold) / len(cold) / 1000.0) if cold else 3.0
        exec_s = max(
            sum(row.exec_ms for row in rows) / len(rows) / 1000.0, 0.001
        )
        best, why = recommend(
            gaps,
            target_cold_rate=target_cold_rate,
            cold_start_s=cold_start_s,
            exec_s=exec_s,
            memory_mb=spec.memory_mb,
        )
        table = sweep(
            gaps, cold_start_s=cold_start_s, exec_s=exec_s, memory_mb=spec.memory_mb
        )
        return {
            "function": name,
            "observed": {
                "requests": len(rows),
                "measured_cold_start_s": round(cold_start_s, 3),
                "mean_exec_s": round(exec_s, 4),
                "current_idle_timeout_s": spec.idle_timeout_s,
            },
            "recommended": best.to_dict(),
            "explanation": why,
            "sweep": [row.to_dict() for row in table],
        }

    # -- console -------------------------------------------------------

    @app.get("/")
    def console() -> FileResponse:
        index = WEB / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="console not built")
        return FileResponse(index)

    if WEB.is_dir():
        app.mount("/web", StaticFiles(directory=WEB), name="web")

    app.state.platform = platform
    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
