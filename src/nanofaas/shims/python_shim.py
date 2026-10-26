"""The process that runs inside a function container.

It does three things and nothing else: import the handler once, answer a
readiness probe, and run the handler per invocation.

**Standard library only, on purpose.** The shim has no dependencies, so
building a function image is a COPY of two files onto a base image -- no
package install, no network at build time, and a build that finishes in under
a second after the first one. A shim that needed Flask would put a pip
install in the critical path of every deploy and several hundred milliseconds
into every cold start.

**The handler is imported once.** That import is the expensive part of a cold
start for any real function -- it is where a model loads or a client connects
-- and doing it at container start rather than per invocation is the entire
reason a warm invocation costs milliseconds.

**Readiness is reported after the import, not before.** The platform's probe
is what it times a cold start against, so answering it before the handler is
importable would report a container as ready while the first invocation was
still waiting for the import, and move that cost somewhere it cannot be seen.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("NANOFAAS_PORT", "8080"))
ENTRYPOINT = os.environ.get("NANOFAAS_ENTRYPOINT", "handle")
HANDLER_PATH = os.environ.get("NANOFAAS_HANDLER", "/app/handler.py")

BOOTED_AT = time.time()
_handler = None
_import_error: str | None = None
_import_ms = 0.0
_invocations = 0


def load_handler() -> None:
    """Import the handler module and find the entrypoint.

    A failure here is recorded rather than raised: the process stays up and
    reports the problem on every request, which turns "my function is broken"
    into a readable error instead of a container that exits and is retried
    until the platform gives up on it.
    """
    global _handler, _import_error, _import_ms
    started = time.perf_counter()
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("nanofaas_handler", HANDLER_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load a module from {HANDLER_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["nanofaas_handler"] = module
        spec.loader.exec_module(module)
        candidate = getattr(module, ENTRYPOINT, None)
        if candidate is None:
            raise AttributeError(f"{HANDLER_PATH} defines no {ENTRYPOINT!r}")
        if not callable(candidate):
            raise TypeError(f"{ENTRYPOINT!r} is not callable")
        _handler = candidate
    except BaseException:
        _import_error = traceback.format_exc(limit=6)
    _import_ms = (time.perf_counter() - started) * 1000.0


class Shim(BaseHTTPRequestHandler):
    # The default logger writes a line per request to stderr, which on a
    # benchmark of 10,000 invocations is both noise and measurable overhead.
    def log_message(self, *args) -> None:  # noqa: D102
        return

    protocol_version = "HTTP/1.1"
    # TCP_NODELAY on the connection. Without it the kernel holds the small
    # second write of a response waiting for an ACK the client is itself
    # delaying, and every invocation pays a flat penalty for nothing. The
    # giveaway was that the cost was identical on every call regardless of
    # what the handler did: 48ms per warm invocation before, 3.8ms after,
    # for a handler whose own execution time is 0.007ms.
    disable_nagle_algorithm = True

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/health", ""):
            self._send(
                200 if _import_error is None else 503,
                {
                    "ready": _import_error is None,
                    "error": _import_error,
                    "import_ms": round(_import_ms, 3),
                    "uptime_s": round(time.time() - BOOTED_AT, 3),
                    "invocations": _invocations,
                },
            )
            return
        self._send(404, {"error": f"no route {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        global _invocations
        if self.path.rstrip("/") != "/invoke":
            self._send(404, {"error": f"no route {self.path}"})
            return
        if _import_error is not None:
            self._send(503, {"error": "handler failed to import", "detail": _import_error})
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            event = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            self._send(400, {"error": f"invalid JSON body: {exc}"})
            return

        started = time.perf_counter()
        try:
            result = _handler(event)
            exec_ms = (time.perf_counter() - started) * 1000.0
            _invocations += 1
            try:
                json.dumps(result)
            except (TypeError, ValueError):
                # A handler returning something unserialisable is a function
                # bug, and saying so beats a 500 from the JSON encoder.
                self._send(
                    500,
                    {
                        "error": "the handler returned a value that is not JSON",
                        "type": type(result).__name__,
                        "exec_ms": round(exec_ms, 3),
                    },
                )
                return
            self._send(200, {"result": result, "exec_ms": round(exec_ms, 3)})
        except BaseException:
            exec_ms = (time.perf_counter() - started) * 1000.0
            _invocations += 1
            self._send(
                500,
                {
                    "error": "the handler raised",
                    "detail": traceback.format_exc(limit=6),
                    "exec_ms": round(exec_ms, 3),
                },
            )


def main() -> None:
    load_handler()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Shim)
    # Daemon threads, so a hung invocation cannot keep the process alive
    # after the platform asks it to stop.
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
