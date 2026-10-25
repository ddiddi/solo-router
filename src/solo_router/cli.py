"""Command line: `solo-router serve | route | task | models | export`."""

from __future__ import annotations

import argparse
import json
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

    p_route = sub.add_parser("route", help="route one instruction and print the decision")
    p_task = sub.add_parser("task", help="route one instruction through perceive -> reason -> act")
    for p in (p_route, p_task):
        _request_args(p)

    sub.add_parser("models", help="list configured models by role")
    sub.add_parser("export", help="print the config as JSON (used by the website playground)")

    args = parser.parse_args(argv)
    router = _router(args.config)

    if args.cmd == "serve":
        from .server import serve

        listener = router.config.listener
        serve(router, args.host or listener.get("address", "0.0.0.0"), args.port or listener.get("port", 8899),
              dry_run=False if args.proxy else None)
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
                    print(f"  {m.name:<20} {m.kind:<9} {m.location:<8} {m.p50_latency_ms:>6g}ms  q={m.quality:.2f}{chunk}")
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
    p.add_argument("--json", action="store_true")


def _request(args: argparse.Namespace) -> RobotRequest:
    return RobotRequest(
        instruction=args.instruction, embodiment=args.embodiment,
        observations=[o for o in args.obs.split(",") if o], control_hz=args.hz,
        latency_budget_ms=args.budget_ms, network=args.network, battery_pct=args.battery,
        humans_nearby=args.humans, preference=args.prefer,
    )


def _print_route(result: RouteResult, indent: str = "") -> None:
    print(f"{indent}{result.decision} -> {result.model} @ {result.location} (~{result.est_latency_ms:g}ms)")
    if result.guardrails:
        print(f"{indent}guardrails: {result.guardrails}")
    if result.fallbacks:
        print(f"{indent}fallbacks: {' -> '.join(result.fallbacks)}")


if __name__ == "__main__":
    raise SystemExit(main())
