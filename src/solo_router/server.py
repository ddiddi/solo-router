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
ANTHROPIC_VERSION = "2023-06-01"


def _decision_headers(result: RouteResult) -> dict[str, str]:
    return {
        "x-solo-router-decision": result.decision,
        "x-solo-router-model": result.model,
        "x-solo-router-location": result.location,
        "x-solo-router-safety": result.signals["safety"],
    }


def _post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _bearer(key: str | None) -> dict[str, str]:
    return {"authorization": f"Bearer {key}"} if key else {}


def _to_anthropic(body: dict[str, Any], model: str) -> dict[str, Any]:
    """OpenAI chat body -> Anthropic Messages body (text and data-URL images)."""
    system, messages = [], []
    for msg in body.get("messages", []):
        content = msg.get("content")
        if msg.get("role") == "system":
            system.append(content if isinstance(content, str) else
                          " ".join(p.get("text", "") for p in content or [] if p.get("type") == "text"))
            continue
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "text":
                    parts.append({"type": "text", "text": part.get("text", "")})
                elif part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if url.startswith("data:") and ";base64," in url:
                        media, data = url[5:].split(";base64,", 1)
                        parts.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
                    elif url:
                        parts.append({"type": "image", "source": {"type": "url", "url": url}})
            content = parts
        messages.append({"role": "assistant" if msg.get("role") == "assistant" else "user", "content": content or ""})
    out: dict[str, Any] = {"model": model, "messages": messages,
                           "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens") or 1024}
    if system:
        out["system"] = "\n\n".join(system)
    for k in ("temperature", "top_p"):
        if k in body:
            out[k] = body[k]
    if body.get("stop"):
        out["stop_sequences"] = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]
    return out


def _from_anthropic(resp: dict[str, Any]) -> dict[str, Any]:
    """Anthropic Messages response -> OpenAI chat.completion."""
    text = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
    usage = resp.get("usage") or {}
    stop = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length"}.get(resp.get("stop_reason"), "stop")
    return {
        "id": resp.get("id") or f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": resp.get("model"),
        "choices": [{"index": 0, "finish_reason": stop, "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": usage.get("input_tokens", 0), "completion_tokens": usage.get("output_tokens", 0),
                  "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)},
    }


class App:
    def __init__(self, router: Router, dry_run: bool | None = None, timeout: float = 30.0,
                 keys: dict[str, str] | None = None):
        self.router = router
        self.dry_run = router.config.listener.get("dry_run", True) if dry_run is None else dry_run
        self.timeout = timeout
        # Provider API keys (BYOK): from each provider's api_key_env, overridden by `keys`.
        # A request can also bring its own with an `x-<provider>-api-key` header.
        self.keys = {name: os.environ[p.api_key_env] for name, p in router.config.providers.items()
                     if os.environ.get(p.api_key_env)}
        self.keys.update({k: v for k, v in (keys or {}).items() if v})

    def _keys(self, headers: dict[str, str]) -> dict[str, str]:
        keys = dict(self.keys)
        for name in self.router.config.providers:
            if headers.get(f"x-{name}-api-key"):
                keys[name] = headers[f"x-{name}-api-key"]
        return keys

    def _request(self, req: RobotRequest, keys: dict[str, str]) -> RobotRequest:
        # A proxy can only call providers it has a key for, so it routes among those unless the
        # request narrows them itself. A dry run routes as if every provider were enabled.
        if req.providers is None and not self.dry_run:
            req.providers = [p for p in self.router.config.providers if p in keys]
        elif req.providers is not None and not self.dry_run:
            req.providers = [p for p in req.providers if p in keys]
        return req

    def _key_for(self, spec: Any, keys: dict[str, str]) -> str | None:
        if spec.provider:
            return keys.get(spec.provider)
        return os.environ.get(spec.api_key_env) if spec.api_key_env else None

    def models(self, headers: dict[str, str] | None = None) -> dict[str, Any]:
        keys = self._keys(headers or {})
        data = [{"id": AUTO_MODEL, "object": "model", "owned_by": "solo-router"}]
        data += [
            {"id": m.name, "object": "model", "owned_by": m.provider or m.location, "role": m.role,
             "kind": m.kind, "location": m.location, "provider": m.provider,
             **({"configured": m.provider in keys} if m.provider else {})}
            for m in self.router.config.models.values()
        ]
        return {"object": "list", "data": data}

    def providers(self, headers: dict[str, str] | None = None) -> dict[str, Any]:
        keys = self._keys(headers or {})
        return {"object": "list", "data": [
            {"id": p.name, "label": p.label, "base_url": p.base_url, "api_format": p.api_format,
             "api_key_env": p.api_key_env, "configured": p.name in keys,
             "models": [m.name for m in self.router.config.models.values() if m.provider == p.name]}
            for p in self.router.config.providers.values()
        ]}

    def route(self, body: dict[str, Any], headers: dict[str, str]) -> tuple[int, dict[str, Any], dict[str, str]]:
        result = self.router.route(self._request(RobotRequest.from_dict(body), self._keys(headers)))
        return 200, result.to_dict(), _decision_headers(result)

    def task(self, body: dict[str, Any], headers: dict[str, str]) -> tuple[int, dict[str, Any], dict[str, str]]:
        task = self.router.route_task(self._request(RobotRequest.from_dict(body), self._keys(headers)))
        out_headers = {f"x-solo-router-{stage}": route.model for stage, route in task.stages}
        return 200, task.to_dict(), out_headers

    def _complete(self, spec: Any, body: dict[str, Any], key: str | None) -> dict[str, Any]:
        upstream = {k: v for k, v in body.items() if k not in ("robot", "stream")}
        base = spec.base_url.rstrip("/")
        if spec.api_format == "anthropic":
            resp = _post(base + "/messages", _to_anthropic(upstream, spec.upstream_id),
                         {"x-api-key": key or "", "anthropic-version": ANTHROPIC_VERSION}, self.timeout)
            return _from_anthropic(resp)
        upstream["model"] = spec.upstream_id
        return _post(base + "/chat/completions", upstream, _bearer(key), self.timeout)

    def chat(self, body: dict[str, Any], headers: dict[str, str]) -> tuple[int, dict[str, Any], dict[str, str]]:
        keys = self._keys(headers)
        req = self._request(RobotRequest.from_openai(body), keys)
        result = self.router.route(req)
        out_headers = _decision_headers(result)
        models = self.router.config.models
        # Like OpenRouter: try the selected model, then each fallback that speaks a chat API.
        chain = [models[n] for n in [result.model, *result.fallbacks]
                 if models[n].base_url and models[n].api_format in ("openai", "anthropic")]

        if not self.dry_run and chain and result.signals["safety"] != "critical":
            errors = []
            for spec in chain:
                try:
                    resp = self._complete(spec, body, self._key_for(spec, keys))
                except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                    detail = exc.read().decode(errors="replace")[:300] if isinstance(exc, urllib.error.HTTPError) else ""
                    errors.append({"model": spec.name, "error": f"{exc} {detail}".strip()})
                    continue
                resp["model"] = spec.name
                resp["solo_router"] = {**result.to_dict(), "served_by": spec.name, "attempts": errors}
                return 200, resp, {**out_headers, "x-solo-router-model": spec.name,
                                   "x-solo-router-location": spec.location}
            return 502, {"error": {"message": "every backend in the fallback chain failed",
                                   "attempts": errors, "fallbacks": result.fallbacks}}, out_headers

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
        }, out_headers

    def actions(self, body: dict[str, Any], headers: dict[str, str]) -> tuple[int, dict[str, Any], dict[str, str]]:
        """Robot-native endpoint: route, then pass observations to the selected policy server."""
        keys = self._keys(headers)
        result = self.router.route(self._request(RobotRequest.from_dict(body), keys))
        headers = _decision_headers(result)
        spec = self.router.config.models[result.model]
        out: dict[str, Any] = {"route": result.to_dict()}
        if result.signals["safety"] == "critical":
            out["action"] = {"type": "stop", "hold_position": True}
        elif not self.dry_run and spec.base_url and spec.kind in ("vla", "nav"):
            payload = {"instruction": body.get("instruction", ""), "observation": body.get("observation"),
                       "guardrails": result.guardrails}
            try:
                out["action"] = _post(spec.base_url.rstrip("/") + "/act", payload,
                                      _bearer(self._key_for(spec, keys)), self.timeout)
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
            allowed = ["content-type", "authorization", *(f"x-{p}-api-key" for p in app.router.config.providers)]
            self.send_header("access-control-allow-headers", ", ".join(allowed))
            self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            headers = {k.lower(): v for k, v in self.headers.items()}
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._send(200, app.models(headers))
            elif self.path == "/v1/providers":
                self._send(200, app.providers(headers))
            else:
                self._send(404, {"error": {"message": f"no route for GET {self.path}"}})

        def do_POST(self) -> None:  # noqa: N802
            routes = {"/v1/route": app.route, "/v1/chat/completions": app.chat, "/v1/actions": app.actions,
                      "/v1/task": app.task}
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
            self._send(*handler(body, {k.lower(): v for k, v in self.headers.items()}))

        def log_message(self, fmt: str, *args: Any) -> None:
            if os.environ.get("SOLO_ROUTER_QUIET") != "1":
                super().log_message(fmt, *args)

    return Handler


def serve(router: Router, host: str, port: int, dry_run: bool | None = None) -> None:
    app = App(router, dry_run=dry_run)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    mode = "dry-run" if app.dry_run else "proxy"
    keyed = ", ".join(app.keys) or "none"
    print(f"solo-router listening on http://{host}:{port} ({mode}; provider keys: {keyed})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
