"""HTTP front end: OpenAI-compatible chat plus robot-native routing endpoints.

Standard library only, so the router runs on a robot without extra packages.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .router import RouteResult, Router
from .signals import RobotRequest

AUTO_MODEL = "solo/auto"


def _decision_headers(result: RouteResult) -> dict[str, str]:
    return {
        "x-solo-router-decision": result.decision,
        "x-solo-router-model": result.model,
        "x-solo-router-location": result.location,
        "x-solo-router-safety": result.signals["safety"],
    }


def _forward(url: str, body: dict[str, Any], api_key_env: str | None, timeout: float) -> dict[str, Any]:
    headers = {"content-type": "application/json"}
    if api_key_env and os.environ.get(api_key_env):
        headers["authorization"] = f"Bearer {os.environ[api_key_env]}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class App:
    def __init__(self, router: Router, dry_run: bool | None = None, timeout: float = 30.0):
        self.router = router
        self.dry_run = router.config.listener.get("dry_run", True) if dry_run is None else dry_run
        self.timeout = timeout

    def models(self) -> dict[str, Any]:
        data = [{"id": AUTO_MODEL, "object": "model", "owned_by": "solo-router"}]
        data += [
            {"id": m.name, "object": "model", "owned_by": m.location, "kind": m.kind}
            for m in self.router.config.models.values()
        ]
        return {"object": "list", "data": data}

    def route(self, body: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, str]]:
        result = self.router.route(RobotRequest.from_dict(body))
        return 200, result.to_dict(), _decision_headers(result)

    def chat(self, body: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, str]]:
        req = RobotRequest.from_openai(body)
        result = self.router.route(req)
        headers = _decision_headers(result)
        spec = self.router.config.models[result.model]

        if not self.dry_run and spec.base_url and spec.api_format == "openai":
            upstream = {k: v for k, v in body.items() if k != "robot"}
            upstream["model"] = spec.name
            try:
                resp = _forward(spec.base_url.rstrip("/") + "/chat/completions", upstream,
                                spec.api_key_env, self.timeout)
            except (urllib.error.URLError, TimeoutError) as exc:
                return 502, {"error": {"message": f"backend {spec.name} unreachable: {exc}",
                                       "fallbacks": result.fallbacks}}, headers
            resp["solo_router"] = result.to_dict()
            return 200, resp, headers

        summary = (f"[solo-router] {result.decision} -> {result.model} ({result.location}, "
                   f"~{result.est_latency_ms:g}ms)")
        if result.guardrails:
            summary += f" guardrails={json.dumps(result.guardrails)}"
        return 200, {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": result.model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": summary}}],
            "solo_router": result.to_dict(),
        }, headers

    def actions(self, body: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, str]]:
        """Robot-native endpoint: route, then pass observations to the selected policy server."""
        result = self.router.route(RobotRequest.from_dict(body))
        headers = _decision_headers(result)
        spec = self.router.config.models[result.model]
        out: dict[str, Any] = {"route": result.to_dict()}
        if result.signals["safety"] == "critical":
            out["action"] = {"type": "stop", "hold_position": True}
        elif not self.dry_run and spec.base_url and spec.kind in ("vla", "nav"):
            payload = {"instruction": body.get("instruction", ""), "observation": body.get("observation"),
                       "guardrails": result.guardrails}
            try:
                out["action"] = _forward(spec.base_url.rstrip("/") + "/act", payload,
                                         spec.api_key_env, self.timeout)
            except (urllib.error.URLError, TimeoutError) as exc:
                out["action"] = {"type": "hold", "reason": f"policy {spec.name} unreachable: {exc}"}
                return 502, out, headers
        return 200, out, headers


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "solo-router/0.1"

        def _send(self, status: int, payload: dict[str, Any], headers: dict[str, str] | None = None) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.send_header("access-control-allow-origin", "*")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.send_header("access-control-allow-origin", "*")
            self.send_header("access-control-allow-headers", "content-type, authorization")
            self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._send(200, app.models())
            else:
                self._send(404, {"error": {"message": f"no route for GET {self.path}"}})

        def do_POST(self) -> None:  # noqa: N802
            routes = {"/v1/route": app.route, "/v1/chat/completions": app.chat, "/v1/actions": app.actions}
            handler = routes.get(self.path)
            if handler is None:
                self._send(404, {"error": {"message": f"no route for POST {self.path}"}})
                return
            try:
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError as exc:
                self._send(400, {"error": {"message": f"invalid JSON: {exc}"}})
                return
            self._send(*handler(body))

        def log_message(self, fmt: str, *args: Any) -> None:
            if os.environ.get("SOLO_ROUTER_QUIET") != "1":
                super().log_message(fmt, *args)

    return Handler


def serve(router: Router, host: str, port: int, dry_run: bool | None = None) -> None:
    app = App(router, dry_run=dry_run)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    mode = "dry-run" if app.dry_run else "proxy"
    print(f"solo-router listening on http://{host}:{port} ({mode})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
