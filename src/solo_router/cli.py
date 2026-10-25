"""Command line: `solo-router serve | route | models | export`."""

from __future__ import annotations

import argparse
import json
import sys

from .config import default_config_path, load_config
from .router import Router
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
    p_route.add_argument("instruction")
    p_route.add_argument("-e", "--embodiment")
    p_route.add_argument("-o", "--obs", default="", help="comma-separated observations, e.g. rgb,proprio")
    p_route.add_argument("--hz", type=float, help="control loop rate")
    p_route.add_argument("--budget-ms", type=float)
    p_route.add_argument("--network", default="online", choices=["online", "degraded", "offline"])
    p_route.add_argument("--battery", type=float)
    p_route.add_argument("--humans", action="store_true", help="humans are near the robot")
    p_route.add_argument("--prefer", default="balanced", choices=["balanced", "quality", "latency", "cost"])
    p_route.add_argument("--json", action="store_true")

    sub.add_parser("models", help="list configured models")
    sub.add_parser("export", help="print the config as JSON (used by the website playground)")

    args = parser.parse_args(argv)
    router = _router(args.config)

    if args.cmd == "serve":
        from .server import serve

        listener = router.config.listener
        serve(router, args.host or listener.get("address", "0.0.0.0"), args.port or listener.get("port", 8899),
              dry_run=False if args.proxy else None)
    elif args.cmd == "route":
        req = RobotRequest(
            instruction=args.instruction, embodiment=args.embodiment,
            observations=[o for o in args.obs.split(",") if o], control_hz=args.hz,
            latency_budget_ms=args.budget_ms, network=args.network, battery_pct=args.battery,
            humans_nearby=args.humans, preference=args.prefer,
        )
        result = router.route(req)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"{result.decision} -> {result.model} @ {result.location} (~{result.est_latency_ms:g}ms)")
            if result.guardrails:
                print(f"guardrails: {result.guardrails}")
            if result.fallbacks:
                print(f"fallbacks: {' -> '.join(result.fallbacks)}")
            for line in result.trace:
                print(f"  · {line}")
    elif args.cmd == "models":
        for m in router.config.models.values():
            print(f"{m.name:<20} {m.kind:<8} {m.location:<8} {m.p50_latency_ms:>6g}ms  q={m.quality:.2f}")
    elif args.cmd == "export":
        json.dump(router.config.raw, sys.stdout, indent=2)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
