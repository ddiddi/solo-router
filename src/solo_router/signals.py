"""Turn a robot request into routing signals."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RobotRequest:
    instruction: str = ""
    embodiment: str | None = None
    # Which observation streams accompany the request: rgb, depth, wrist_rgb, proprio, lidar, audio, force.
    observations: list[str] = field(default_factory=list)
    control_hz: float | None = None
    latency_budget_ms: float | None = None
    network: str = "online"  # online | degraded | offline
    battery_pct: float | None = None
    humans_nearby: bool = False
    estop: bool = False
    preference: str = "balanced"  # quality | latency | cost | balanced
    model: str = "solo/auto"
    # Where this client can reach compute. None means onboard, edge and cloud; add "browser"
    # when the caller can run ONNX models itself, and list its runtimes in client_runtimes.
    locations: list[str] | None = None
    client_runtimes: list[str] = field(default_factory=list)
    # Cloud providers the caller has API keys for, e.g. ["openai", "nebius"]. None means every
    # configured provider; [] means none, so only self-hosted models are routable.
    providers: list[str] | None = None
    # Signals an orchestrator already knows, e.g. {"intent": "perception"} for a perception stage.
    # Only intent and horizon can be overridden; safety is always computed from the request.
    overrides: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobotRequest":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @classmethod
    def from_openai(cls, body: dict[str, Any]) -> "RobotRequest":
        """Build a request from an OpenAI chat body plus an optional `robot` extension object."""
        text, has_image = "", False
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif part.get("type") in ("image_url", "input_image", "image"):
                        has_image = True
                text = " ".join(parts)
            break
        req = cls.from_dict(body.get("robot") or {})
        req.instruction = req.instruction or text
        req.model = body.get("model", req.model)
        if has_image and "rgb" not in req.observations:
            req.observations = [*req.observations, "rgb"]
        return req


@dataclass
class Signals:
    intent: str
    intent_scores: dict[str, int]
    horizon: str  # short | long
    safety: str  # normal | elevated | critical
    safety_reasons: list[str]
    latency_budget_ms: float | None
    latency_tier: str  # realtime | interactive | deliberative
    modalities: list[str]
    embodiment: str | None
    network: str
    energy: str  # normal | low
    preference: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _phrase_regex(phrase: str) -> re.Pattern[str]:
    return re.compile(r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])")


class SignalExtractor:
    def __init__(self, cfg: dict[str, Any]):
        self.intents = {
            name: [(kw, _phrase_regex(kw)) for kw in words]
            for name, words in cfg.get("intent", {}).items()
        }
        horizon = cfg.get("horizon", {})
        self.long_markers = horizon.get("long_markers", 2)
        self.long_words = horizon.get("long_words", 24)
        self.markers = [_phrase_regex(m) for m in horizon.get("markers", [])]
        safety = cfg.get("safety", {})
        self.critical = [re.compile(p, re.IGNORECASE) for p in safety.get("critical_patterns", [])]
        self.hazards = [(w, _phrase_regex(w)) for w in safety.get("hazard_words", [])]
        latency = cfg.get("latency", {})
        self.realtime_ms = latency.get("realtime_ms", 50)
        self.interactive_ms = latency.get("interactive_ms", 500)
        self.low_battery = cfg.get("energy", {}).get("low_battery_pct", 20)

    def extract(self, req: RobotRequest) -> Signals:
        text = req.instruction.lower().strip()

        scores = {name: sum(1 for _, rx in kws if rx.search(text)) for name, kws in self.intents.items()}
        best = max(scores.values(), default=0)
        intent = next((n for n, s in scores.items() if s == best), "conversation") if best else "conversation"

        marker_hits = sum(len(rx.findall(text)) for rx in self.markers) + text.count(",")
        long = marker_hits >= self.long_markers or len(text.split()) >= self.long_words
        horizon = "long" if long else "short"

        intent = req.overrides.get("intent", intent)
        horizon = req.overrides.get("horizon", horizon)

        safety, reasons = "normal", []
        if req.estop:
            safety, reasons = "critical", ["estop flag set"]
        elif any(rx.search(text) for rx in self.critical):
            safety, reasons = "critical", ["stop command in instruction"]
        else:
            hazards = [w for w, rx in self.hazards if rx.search(text)]
            if hazards:
                reasons.append("hazard words: " + ", ".join(hazards))
            if req.humans_nearby and intent in ("manipulation", "navigation"):
                reasons.append(f"humans nearby during {intent}")
            if reasons:
                safety = "elevated"

        budget = req.latency_budget_ms
        if budget is None and req.control_hz:
            budget = 1000.0 / req.control_hz
        if budget is not None and budget < self.realtime_ms:
            tier = "realtime"
        elif budget is not None and budget < self.interactive_ms:
            tier = "interactive"
        else:
            tier = "deliberative"

        energy = "low" if req.battery_pct is not None and req.battery_pct < self.low_battery else "normal"

        return Signals(
            intent=intent,
            intent_scores=scores,
            horizon=horizon,
            safety=safety,
            safety_reasons=reasons,
            latency_budget_ms=budget,
            latency_tier=tier,
            modalities=["text", *req.observations] if text else list(req.observations),
            embodiment=req.embodiment,
            network=req.network,
            energy=energy,
            preference=req.preference,
        )
