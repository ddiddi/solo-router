"""Command line: `solo-router run | serve | route | task | models | providers | export`."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .config import ROLES, default_config_path, load_config
from .router import RouteResult, Router
from .signals import RobotRequest


def _router(path: str | None) -> Router:
    return Router(load_config(path or default_config_path()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="solo-router", description="Semantic router for robots.")
    parser.add_argument("-c", "--config", help="path to router.yaml (default: bundled config)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="run the HTTP router")
    p_serve.add_argument("--host")
    p_serve.add_argument("--port", type=int)
    p_serve.add_argument("--proxy", action="store_true", help="forward to backends instead of dry-run")

    p_run = sub.add_parser("run", help="route one prompt and call the chosen model with your API keys")
    p_route = sub.add_parser("route", help="route one instruction and print the decision")
    p_task = sub.add_parser("task", help="route one instruction through perceive -> reason -> act")
    for p in (p_run, p_route, p_task):
        _request_args(p)
    p_run.add_argument("--local", action="store_true",
                       help="also route to self-hosted onboard/edge endpoints, not only hosted providers")

    sub.add_parser("models", help="list configured models by role")
    sub.add_parser("providers", help="list cloud providers and whether their API key is set")
    sub.add_parser("export", help="print the config as JSON (used by the website playground)")

    args = parser.parse_args(argv)
    router = _router(args.config)

    if args.cmd == "serve":
        from .server import serve

        listener = router.config.listener
        serve(router, args.host or listener.get("address", "0.0.0.0"), args.port or listener.get("port", 8899),
              dry_run=False if args.proxy else None)
    elif args.cmd == "run":
        return _run(router, args)
    elif args.cmd == "route":
        result = router.route(_request(args))
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            _print_route(result)
            for line in result.trace:
                print(f"  · {line}")
    elif args.cmd == "task":
        task = router.route_task(_request(args))
        if args.json:
            print(json.dumps(task.to_dict(), indent=2))
        else:
            for stage, result in task.stages:
                print(f"[{stage}]")
                _print_route(result, indent="  ")
    elif args.cmd == "models":
        for role in ROLES:
            print(role)
            for m in router.config.models.values():
                if m.role == role:
                    chunk = f" chunk={m.action_chunk}" if m.action_chunk > 1 else ""
                    print(f"  {m.name:<28} {m.kind:<10} {m.location:<8} {m.p50_latency_ms:>6g}ms  q={m.quality:.2f}{chunk}")
    elif args.cmd == "providers":
        for p in router.config.providers.values():
            status = "key set" if os.environ.get(p.api_key_env) else "no key"
            print(f"{p.name:<10} {p.label:<22} ${p.api_key_env:<18} {status}")
            for m in router.config.models.values():
                if m.provider == p.name:
                    print(f"  {m.name:<28} -> {m.upstream_id}")
    elif args.cmd == "export":
        json.dump(router.config.raw, sys.stdout, indent=2)
        print()
    return 0


def _request_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("instruction")
    p.add_argument("-e", "--embodiment")
    p.add_argument("-o", "--obs", default="", help="comma-separated observations, e.g. rgb,proprio")
    p.add_argument("--hz", type=float, help="control loop rate")
    p.add_argument("--budget-ms", type=float)
    p.add_argument("--network", default="online", choices=["online", "degraded", "offline"])
    p.add_argument("--battery", type=float)
    p.add_argument("--humans", action="store_true", help="humans are near the robot")
    p.add_argument("--prefer", default="balanced", choices=["balanced", "quality", "latency", "cost"])
    p.add_argument("--providers", help="cloud providers you have keys for, e.g. openai,nebius ('' for none); "
                                       "default: all")
    p.add_argument("--json", action="store_true")


def _request(args: argparse.Namespace) -> RobotRequest:
    return RobotRequest(
        instruction=args.instruction, embodiment=args.embodiment,
        observations=[o for o in args.obs.split(",") if o], control_hz=args.hz,
        latency_budget_ms=args.budget_ms, network=args.network, battery_pct=args.battery,
        humans_nearby=args.humans, preference=args.prefer,
        providers=None if args.providers is None else [p for p in args.providers.split(",") if p],
    )


def _run(router: Router, args: argparse.Namespace) -> int:
    """One prompt, one command: route among the providers you have keys for, then call the model."""
    from .server import App

    app = App(router, dry_run=False)
    if not app.keys and not args.local:
        print("no provider key found: export OPENAI_API_KEY, ANTHROPIC_API_KEY or NEBIUS_API_KEY "
              "(or pass --local to use self-hosted endpoints)", file=sys.stderr)
        return 1
    req = _request(args)
    robot = {k: v for k, v in vars(req).items() if k not in ("instruction", "model", "overrides")}
    if not args.local:
        robot["locations"] = ["cloud"]
    body = {"model": "solo/auto", "robot": robot, "messages": [{"role": "user", "content": args.instruction}],
            "stream": not args.json}
    status, resp, headers = app.chat(body, {})
    if status == 200 and not isinstance(resp, dict):
        return _print_stream(resp, headers)
    if args.json:
        print(json.dumps(resp, indent=2))
        return 0 if status == 200 else 1
    if status != 200:
        print(f"error: {resp['error']['message']}", file=sys.stderr)
        for attempt in resp["error"].get("attempts", []):
            print(f"  {attempt['model']}: {attempt['error']}", file=sys.stderr)
        return 1
    info = resp["solo_router"]
    served = info.get("served_by", info["model"])
    location = router.config.models[served].location
    print(f"[{info['decision']} -> {served} @ {location}]", file=sys.stderr)
    print(resp["choices"][0]["message"]["content"])
    return 0


def _print_stream(events, headers: dict[str, str]) -> int:
    print(f"[{headers['x-solo-router-decision']} -> {headers['x-solo-router-model']} "
          f"@ {headers['x-solo-router-location']}]", file=sys.stderr, flush=True)
    for event in events:
        data = event.decode().removeprefix("data: ").strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if "error" in chunk:
            print(f"\nerror: {chunk['error']['message']}", file=sys.stderr)
            return 1
        for choice in chunk.get("choices", []):
            print(choice["delta"].get("content") or "", end="", flush=True)
    print()
    return 0


def _print_route(result: RouteResult, indent: str = "") -> None:
    print(f"{indent}{result.decision} -> {result.model} @ {result.location} (~{result.est_latency_ms:g}ms)")
    if result.guardrails:
        print(f"{indent}guardrails: {result.guardrails}")
    if result.fallbacks:
        print(f"{indent}fallbacks: {' -> '.join(result.fallbacks)}")


if __name__ == "__main__":
    raise SystemExit(main())
