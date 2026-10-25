"""Decisions and selection algorithms."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import Decision, ModelSpec, RouterConfig
from .signals import RobotRequest, SignalExtractor, Signals

FAILSAFE = "safety-controller"

# (quality, latency, cost) weights for the hybrid algorithm.
PREFERENCE_WEIGHTS = {
    "balanced": (1.0, 0.3, 0.05),
    "quality": (1.0, 0.05, 0.02),
    "latency": (0.6, 1.0, 0.05),
    "cost": (0.6, 0.1, 1.0),
}
DEGRADED_CLOUD_PENALTY = 0.3
LOW_BATTERY_PENALTY_PER_W = 0.01
DEFAULT_LATENCY_SCALE_MS = 3000.0
# Observation streams that satisfy a model's "rgb" requirement.
RGB_STREAMS = {"rgb", "wrist_rgb"}


@dataclass
class Candidate:
    model: str
    location: str
    score: float
    p50_latency_ms: float


@dataclass
class RouteResult:
    decision: str
    algorithm: str
    model: str
    location: str
    base_url: str | None
    api_format: str | None
    est_latency_ms: float
    signals: dict[str, Any]
    guardrails: dict[str, Any]
    ranked: list[Candidate]
    fallbacks: list[str]
    rejected: dict[str, str]
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def matches(when: dict[str, Any], signals: Signals) -> bool:
    for key, expected in when.items():
        if key == "any":
            if not any(matches(sub, signals) for sub in expected):
                return False
            continue
        if key == "all":
            if not all(matches(sub, signals) for sub in expected):
                return False
            continue
        actual = getattr(signals, key, None)
        allowed = expected if isinstance(expected, list) else [expected]
        if actual not in allowed:
            return False
    return True


class Router:
    def __init__(self, config: RouterConfig):
        self.config = config
        self.extractor = SignalExtractor(config.signals)
        self.min_elevated_tier = config.guardrails.get("elevated", {}).get("min_safety_tier", 0)

    def route(self, request: RobotRequest | dict[str, Any]) -> RouteResult:
        if isinstance(request, dict):
            request = RobotRequest.from_dict(request)
        signals = self.extractor.extract(request)
        trace = [
            f"signals: intent={signals.intent} horizon={signals.horizon} safety={signals.safety} "
            f"latency={signals.latency_tier} network={signals.network}"
        ]

        decision = next(
            (d for d in self.config.decisions if matches(d.when, signals)),
            Decision("failsafe", 0, {}, [FAILSAFE], "static"),
        )
        trace.append(f"decision: {decision.name} (priority {decision.priority}, {decision.algorithm})")

        feasible, rejected = [], {}
        for name in decision.candidates:
            spec = self.config.models[name]
            why = self._reject_reason(spec, signals, request)
            if why:
                rejected[name] = why
                trace.append(f"reject {name}: {why}")
            else:
                feasible.append(spec)

        ranked = self._rank(decision, feasible, signals)
        if ranked:
            chosen = self.config.models[ranked[0].model]
            trace.append(f"select {chosen.name} @ {chosen.location} (score {ranked[0].score:.3f})")
        else:
            chosen = self.config.models[FAILSAFE]
            trace.append("no feasible candidate: falling back to safety-controller (hold position)")

        fallbacks = [c.model for c in ranked[1:]]
        if chosen.name != FAILSAFE:
            fallbacks.append(FAILSAFE)

        guardrails = {}
        if signals.safety == "elevated":
            guardrails = {k: v for k, v in self.config.guardrails.get("elevated", {}).items()
                          if k != "min_safety_tier"}
        elif signals.safety == "critical":
            guardrails = {"action": "stop", "hold_position": True}

        return RouteResult(
            decision=decision.name,
            algorithm=decision.algorithm,
            model=chosen.name,
            location=chosen.location,
            base_url=chosen.base_url,
            api_format=chosen.api_format,
            est_latency_ms=chosen.p50_latency_ms,
            signals=signals.to_dict(),
            guardrails=guardrails,
            ranked=ranked,
            fallbacks=fallbacks,
            rejected=rejected,
            trace=trace,
        )

    def _reject_reason(self, spec: ModelSpec, s: Signals, req: RobotRequest) -> str | None:
        if spec.name == FAILSAFE:
            return None
        if not spec.supports(s.embodiment):
            return f"does not support embodiment {s.embodiment}"
        for need in spec.requires:
            have = RGB_STREAMS if need == "rgb" else {need}
            if not have & set(s.modalities):
                return f"needs {need} observation"
        if s.network == "offline" and spec.location == "cloud":
            return "robot is offline"
        if s.latency_budget_ms is not None and spec.p50_latency_ms > s.latency_budget_ms:
            return f"p50 {spec.p50_latency_ms:g}ms exceeds budget {s.latency_budget_ms:g}ms"
        if req.control_hz and spec.max_hz is not None and spec.max_hz < req.control_hz:
            return f"max {spec.max_hz:g}Hz below control rate {req.control_hz:g}Hz"
        if s.safety == "elevated" and spec.safety_tier < self.min_elevated_tier:
            return f"safety tier {spec.safety_tier} below required {self.min_elevated_tier}"
        return None

    def _rank(self, decision: Decision, feasible: list[ModelSpec], s: Signals) -> list[Candidate]:
        if not feasible:
            return []
        if decision.algorithm == "static":
            return [Candidate(m.name, m.location, 1.0, m.p50_latency_ms) for m in feasible]
        if decision.algorithm == "latency_aware":
            ordered = sorted(feasible, key=lambda m: (-m.quality, m.p50_latency_ms))
            return [Candidate(m.name, m.location, m.quality, m.p50_latency_ms) for m in ordered]

        # A decision's own preference applies unless the request states one.
        pref = s.preference if s.preference != "balanced" else decision.prefer or "balanced"
        if pref not in PREFERENCE_WEIGHTS:
            pref = "balanced"
        wq, wl, wc = PREFERENCE_WEIGHTS[pref]
        scale = s.latency_budget_ms or DEFAULT_LATENCY_SCALE_MS
        max_cost = max((m.cost_per_1k for m in feasible), default=0) or 1.0
        scored = []
        for m in feasible:
            score = wq * m.quality - wl * min(m.p50_latency_ms / scale, 1.0) - wc * m.cost_per_1k / max_cost
            if s.network == "degraded" and m.location == "cloud":
                score -= DEGRADED_CLOUD_PENALTY
            if s.energy == "low":
                score -= LOW_BATTERY_PENALTY_PER_W * m.power_w
            scored.append(Candidate(m.name, m.location, round(score, 4), m.p50_latency_ms))
        return sorted(scored, key=lambda c: -c.score)
