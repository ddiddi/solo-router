"""Load and validate router configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOCATIONS = ("onboard", "edge", "cloud")
ALGORITHMS = ("static", "latency_aware", "hybrid")


class ConfigError(ValueError):
    pass


@dataclass
class ModelSpec:
    name: str
    kind: str
    location: str
    p50_latency_ms: float
    quality: float
    max_hz: float | None = None
    cost_per_1k: float = 0.0
    power_w: float = 0.0
    embodiments: list[str] = field(default_factory=lambda: ["*"])
    requires: list[str] = field(default_factory=list)
    safety_tier: int = 1
    base_url: str | None = None
    api_format: str | None = None
    api_key_env: str | None = None
    description: str = ""

    def supports(self, embodiment: str | None) -> bool:
        return embodiment is None or "*" in self.embodiments or embodiment in self.embodiments


@dataclass
class Decision:
    name: str
    priority: int
    when: dict[str, Any]
    candidates: list[str]
    algorithm: str
    prefer: str | None = None


@dataclass
class RouterConfig:
    signals: dict[str, Any]
    models: dict[str, ModelSpec]
    decisions: list[Decision]
    guardrails: dict[str, Any]
    listener: dict[str, Any]
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RouterConfig":
        models: dict[str, ModelSpec] = {}
        for entry in data.get("models", []):
            spec = ModelSpec(**entry)
            if spec.location not in LOCATIONS:
                raise ConfigError(f"model {spec.name}: unknown location {spec.location!r}")
            if spec.name in models:
                raise ConfigError(f"duplicate model {spec.name}")
            models[spec.name] = spec

        decisions = []
        for entry in data.get("decisions", []):
            d = Decision(
                name=entry["name"],
                priority=int(entry.get("priority", 0)),
                when=entry.get("when") or {},
                candidates=list(entry["candidates"]),
                algorithm=entry.get("algorithm", "static"),
                prefer=entry.get("prefer"),
            )
            if d.algorithm not in ALGORITHMS:
                raise ConfigError(f"decision {d.name}: unknown algorithm {d.algorithm!r}")
            missing = [c for c in d.candidates if c not in models]
            if missing:
                raise ConfigError(f"decision {d.name}: unknown candidates {missing}")
            decisions.append(d)
        decisions.sort(key=lambda d: d.priority, reverse=True)

        if "safety-controller" not in models:
            raise ConfigError("a model named 'safety-controller' is required as the fail-safe")

        return cls(
            signals=data.get("signals", {}),
            models=models,
            decisions=decisions,
            guardrails=data.get("guardrails", {}),
            listener=data.get("listener", {}),
            raw=data,
        )


def load_config(path: str | Path) -> RouterConfig:
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".json":
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    return RouterConfig.from_dict(data)


def default_config_path() -> Path:
    return Path(__file__).resolve().with_name("router.yaml")
