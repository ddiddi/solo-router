"""The website playground must make the same decisions as the Python router."""

import itertools
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from solo_router import Router, load_config
from solo_router.config import default_config_path

HERE = Path(__file__).resolve().parent

INSTRUCTIONS = [
    "pick up the red cup and place it on the tray", "stop!", "emergency, the arm is stuck",
    "hand me the scissors", "fold the towel", "patrol the warehouse aisle", "go to the kitchen",
    "clean the kitchen, then load the dishwasher and finally wipe the counters",
    "what do you see on the shelf", "why is joint 3 overheating", "hello, who are you", "",
]
EMBODIMENTS = [None, "arm", "so101", "humanoid", "quadruped", "drone"]
CONTEXTS = [
    {"observations": ["rgb", "proprio"], "control_hz": 20},
    {"observations": [], "network": "offline"},
    {"observations": ["wrist_rgb"], "control_hz": 100, "humans_nearby": True},
    {"observations": ["rgb"], "network": "degraded", "battery_pct": 10, "preference": "cost"},
    {"observations": ["rgb"], "latency_budget_ms": 400, "preference": "quality", "estop": False},
    {"observations": ["rgb"], "estop": True},
    {"observations": ["rgb"], "control_hz": 30, "overrides": {"intent": "perception", "horizon": "short"}},
    {"observations": ["rgb"], "control_hz": 30, "locations": ["browser"], "client_runtimes": ["webgpu", "wasm"]},
    {"observations": ["rgb"], "locations": ["browser", "edge"], "client_runtimes": []},
    {"observations": ["rgb"], "providers": ["openai"]},
    {"observations": [], "providers": ["nebius", "anthropic"], "network": "degraded"},
    {"observations": ["rgb"], "providers": [], "locations": ["cloud"]},
]
KEYS = ("decision", "model", "provider", "rejected", "fallbacks", "guardrails", "trace", "signals")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_router_matches_python():
    reqs = [{"instruction": i, "embodiment": e, **c}
            for i, e, c in itertools.product(INSTRUCTIONS, EMBODIMENTS, CONTEXTS)]
    tasks = [{"instruction": i, "embodiment": e, **c}
             for i, e, c in itertools.product(INSTRUCTIONS[:6], ["so101", "humanoid"], CONTEXTS)]
    out = subprocess.run(["node", str(HERE / "js_route.mjs")], input=json.dumps({"routes": reqs, "tasks": tasks}),
                         capture_output=True, text=True, check=True)
    js = json.loads(out.stdout)
    router = Router(load_config(default_config_path()))
    for req, got in zip(reqs, js["routes"]):
        py = router.route(req).to_dict()
        for key in KEYS:
            assert got[key] == py[key], (req, key)
    for req, got in zip(tasks, js["tasks"]):
        py = router.route_task(req).to_dict()["stages"]
        assert [s["stage"] for s in got] == [s["stage"] for s in py]
        for g, p in zip(got, py):
            for key in KEYS:
                assert g[key] == p[key], (req, g["stage"], key)
