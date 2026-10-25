import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from solo_router import RobotRequest, Router, load_config
from solo_router.config import ConfigError, RouterConfig, default_config_path
from solo_router.server import App, make_handler

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def router():
    return Router(load_config(default_config_path()))


def route(router, instruction, **kw):
    return router.route(RobotRequest(instruction=instruction, **kw))


@pytest.mark.parametrize("text", ["stop", "STOP!", "emergency, the arm is stuck", "e-stop", "stop moving"])
def test_stop_commands_preempt_everything(router, text):
    r = route(router, text, embodiment="arm", observations=["rgb"])
    assert r.decision == "emergency_stop"
    assert r.model == "safety-controller"
    assert r.guardrails == {"action": "stop", "hold_position": True}


def test_estop_flag_overrides_instruction(router):
    r = route(router, "pick up the cup", estop=True, embodiment="arm", observations=["rgb"])
    assert r.model == "safety-controller"


def test_manipulation_picks_best_vla_for_embodiment(router):
    assert route(router, "pick up the red cup", embodiment="arm", observations=["rgb"], control_hz=20).model == "pi0-3b"
    assert route(router, "fold the towel", embodiment="humanoid", observations=["rgb"], control_hz=20).model == "gr00t-n1.5-3b"


def test_action_chunk_extends_latency_budget(router):
    # 30 Hz is a 33 ms step, but GR00T predicts 16 steps per call.
    r = route(router, "fold the towel", embodiment="humanoid", observations=["rgb"], control_hz=30)
    assert r.model == "gr00t-n1.5-3b"
    r = route(router, "fold the towel", embodiment="humanoid", observations=["rgb"], control_hz=500)
    assert r.rejected["gr00t-n1.5-3b"].startswith("p50 40ms exceeds 16-step chunk budget")


def test_specific_robot_inherits_family_policies(router):
    r = route(router, "put the red cube in the bowl", embodiment="so101", observations=["rgb"], control_hz=30)
    assert r.model == "pi0-3b"
    assert "act-so101" in r.fallbacks and "gr00t-n1.5-3b" in r.rejected
    r = route(router, "pick up the cup", embodiment="koch", observations=["rgb"], control_hz=30)
    assert "act-so101" in r.rejected


def test_overrides_cannot_lower_safety(router):
    r = router.route(RobotRequest(instruction="stop", overrides={"intent": "manipulation", "safety": "normal"}))
    assert r.model == "safety-controller"


def test_task_pipeline_routes_each_role(router):
    task = router.route_task(RobotRequest(instruction="put the red cube in the bowl", embodiment="so101",
                                          observations=["rgb"], control_hz=30))
    roles = {stage: router.config.models[r.model].role for stage, r in task.stages}
    assert roles == {"perceive": "perception", "reason": "reasoning", "act": "action"}
    # The control-rate deadline only binds the stage that moves the robot.
    assert task.stages[0][1].signals["latency_budget_ms"] is None


def test_task_pipeline_offline_stays_local(router):
    task = router.route_task(RobotRequest(instruction="put the red cube in the bowl", embodiment="so101",
                                          observations=["rgb"], control_hz=30, network="offline"))
    assert all(r.location != "cloud" for _, r in task.stages)


def test_browser_models_need_browser_location_and_runtime(router):
    base = dict(instruction="put the red cube in the bowl", embodiment="so101", observations=["rgb"], control_hz=30)
    # By default the browser is not a reachable location.
    r = router.route(RobotRequest(**base, overrides={"intent": "perception"}))
    assert r.rejected["florence-2-base-web"] == "browser not reachable"
    # A browser client with WebGPU gets the ONNX models for every stage.
    task = router.route_task(RobotRequest(**base, locations=["browser"], client_runtimes=["webgpu", "wasm"]))
    assert [r.model for _, r in task.stages] == ["florence-2-base-web", "qwen2.5-0.5b-web", "so101-ik-web"]
    # Without a runtime the ONNX models are rejected and the stage fails safe.
    task = router.route_task(RobotRequest(**base, locations=["browser"], client_runtimes=[]))
    perceive = task.stages[0][1]
    assert perceive.model == "safety-controller"
    assert perceive.rejected["florence-2-base-web"] == "needs runtime webgpu or wasm"


def test_browser_models_declare_how_to_load(router):
    for m in router.config.models.values():
        if m.location == "browser" and m.kind != "controller":
            assert m.onnx and m.onnx["hf_id"] and m.runtimes


def test_elevated_safety_only_restricts_action_models(router):
    r = route(router, "is there a knife on the table", observations=["rgb"])
    assert r.decision == "scene_understanding" and r.signals["safety"] == "elevated"
    assert r.model != "claude-sonnet-5-5"


def test_control_rate_filters_slow_policies(router):
    r = route(router, "pick up the cup", embodiment="arm", observations=["rgb"], control_hz=10, latency_budget_ms=300)
    assert "openvla-7b" in r.rejected
    assert r.rejected["openvla-7b"] == "max 6Hz below control rate 10Hz"


def test_vla_without_camera_is_rejected(router):
    r = route(router, "pick up the cup", embodiment="arm")
    assert r.model == "safety-controller"
    assert r.rejected["pi0-3b"] == "needs rgb observation"


def test_wrist_camera_satisfies_rgb(router):
    assert route(router, "grasp the block", embodiment="arm", observations=["wrist_rgb"]).model == "pi0-3b"


def test_humans_nearby_elevates_and_restricts_tier(router):
    r = route(router, "hand me the cup", embodiment="arm", observations=["rgb"], humans_nearby=True)
    assert r.signals["safety"] == "elevated"
    assert r.guardrails["max_speed_mps"] == 0.25
    assert "smolvla-450m" in r.rejected


def test_navigation_stays_onboard(router):
    r = route(router, "go to the kitchen", embodiment="quadruped", control_hz=20)
    assert (r.decision, r.location) == ("navigation", "onboard")


def test_long_horizon_goes_to_cloud_planner_when_online(router):
    r = route(router, "clean the kitchen, then load the dishwasher and finally wipe the counters")
    assert r.decision == "long_horizon_task"
    assert r.model == "claude-sonnet-5-5"


def test_offline_never_routes_to_cloud(router):
    r = route(router, "clean the kitchen, then load the dishwasher", network="offline")
    assert r.location != "cloud"
    assert r.rejected["claude-sonnet-5-5"] == "robot is offline"


def test_degraded_network_prefers_edge(router):
    r = route(router, "set the table for four, then serve water", network="degraded")
    assert r.model == "qwen3-8b"


def test_tight_budget_excludes_cloud(router):
    r = route(router, "what do you see on the shelf", observations=["rgb"], latency_budget_ms=400)
    assert r.location != "cloud"
    assert "claude-sonnet-5-5" in r.rejected


def test_conversation_and_diagnostics_stay_local(router):
    assert route(router, "hello, who are you").model == "smollm2-1.7b"
    assert route(router, "why is joint 3 overheating").decision == "diagnostics"


def test_fallback_chain_ends_in_safety_controller(router):
    r = route(router, "pick up the red cup", embodiment="arm", observations=["rgb"])
    assert r.fallbacks[-1] == "safety-controller"


def test_config_rejects_unknown_candidates():
    with pytest.raises(ConfigError):
        RouterConfig.from_dict({
            "models": [{"name": "safety-controller", "kind": "safety", "role": "safety", "location": "onboard",
                        "p50_latency_ms": 1, "quality": 1}],
            "decisions": [{"name": "x", "candidates": ["nope"]}],
        })


def test_config_reports_missing_model_fields():
    with pytest.raises(ConfigError, match="role"):
        RouterConfig.from_dict({"models": [{"name": "x", "kind": "vla", "location": "edge",
                                            "p50_latency_ms": 1, "quality": 1}]})


def test_openai_body_with_image_adds_rgb():
    req = RobotRequest.from_openai({
        "model": "solo/auto",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "pick up the cup"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}],
        "robot": {"embodiment": "arm"},
    })
    assert req.instruction == "pick up the cup"
    assert req.observations == ["rgb"] and req.embodiment == "arm"


def test_website_playground_config_in_sync():
    exported = json.loads((ROOT / "website" / "assets" / "router-config.json").read_text())
    assert exported == load_config(default_config_path()).raw, \
        "run: solo-router export > website/assets/router-config.json"


def test_http_server_end_to_end(router):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(App(router, dry_run=True)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        body = json.dumps({"model": "solo/auto", "messages": [{"role": "user", "content": "stop"}]}).encode()
        req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            assert resp.headers["x-solo-router-model"] == "safety-controller"
            payload = json.loads(resp.read())
        assert payload["solo_router"]["decision"] == "emergency_stop"
        with urllib.request.urlopen(base + "/v1/models") as resp:
            assert json.loads(resp.read())["data"][0]["id"] == "solo/auto"
        body = json.dumps({"instruction": "put the red cube in the bowl", "embodiment": "so101",
                           "observations": ["rgb"], "control_hz": 30}).encode()
        with urllib.request.urlopen(urllib.request.Request(base + "/v1/task", data=body)) as resp:
            assert resp.headers["x-solo-router-act"] == "pi0-3b"
            assert [s["stage"] for s in json.loads(resp.read())["stages"]] == ["perceive", "reason", "act"]
    finally:
        httpd.shutdown()
