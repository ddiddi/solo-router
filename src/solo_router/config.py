"""Load and validate router configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# "browser" models run client-side through ONNX Runtime Web (WebGPU, or WASM as a fallback).
LOCATIONS = ("onboard", "edge", "cloud", "browser")
ROLES = ("perception", "reasoning", "action", "safety")
ALGORITHMS = ("static", "latency_aware", "hybrid")
API_FORMATS = ("openai", "anthropic")


class ConfigError(ValueError):
    pass


@dataclass
class ModelSpec:
    name: str
    kind: str
    role: str
    location: str
    p50_latency_ms: float
    quality: float
    max_hz: float | None = None
    # Steps of motion predicted per inference call. A chunked policy only has to
    # answer once per chunk, so its latency budget is chunk * (1000 / control_hz).
    action_chunk: int = 1
    cost_per_1k: float = 0.0
    power_w: float = 0.0
    embodiments: list[str] = field(default_factory=lambda: ["*"])
    requires: list[str] = field(default_factory=list)
    safety_tier: int = 1
    base_url: str | None = None
    api_format: str | None = None
    api_key_env: str | None = None
    # Cloud provider that serves this model (a key of `providers`). Its base_url, api_format and
    # api_key_env fill in any the model leaves unset. A model with a provider is only routable when
    # the request enables that provider, i.e. the caller has an API key for it.
    provider: str | None = None
    # Model ID the provider expects, when it differs from `name` (e.g. openai/gpt-oss-120b on Nebius).
    # Defaults to `name` without its "provider/" prefix.
    upstream_model: str | None = None
    # Client runtimes the model needs, any one of which suffices (e.g. webgpu, wasm). Empty: none.
    runtimes: list[str] = field(default_factory=list)
    # How a browser loads it: {"hf_id": ..., "task": ..., "dtype": ...} for transformers.js.
    onnx: dict[str, Any] | None = None
    description: str = ""

    def supports(self, embodiment: str | None, family: str | None = None) -> bool:
        if embodiment is None or "*" in self.embodiments:
            return True
        return embodiment in self.embodiments or (family is not None and family in self.embodiments)

    @property
    def upstream_id(self) -> str:
        if self.upstream_model:
            return self.upstream_model
        prefix = f"{self.provider}/" if self.provider else ""
        return self.name[len(prefix):] if prefix and self.name.startswith(prefix) else self.name


@dataclass
class ProviderSpec:
    """A hosted model API you bring your own key for (OpenAI, Anthropic, Nebius Token Factory, ...)."""

    name: str
    label: str
    base_url: str
    api_format: str  # openai | anthropic
    api_key_env: str
    console_url: str = ""


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
    # Specific robots mapped to the embodiment family policies are trained for, e.g. so101 -> arm.
    embodiment_families: dict[str, str] = field(default_factory=dict)
    # Ordered stages for /v1/task: each forces some signals and routes independently.
    pipeline: list[dict[str, Any]] = field(default_factory=list)
    providers: dict[str, ProviderSpec] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RouterConfig":
        providers: dict[str, ProviderSpec] = {}
        for name, entry in (data.get("providers") or {}).items():
            try:
                providers[name] = ProviderSpec(name=name, **entry)
            except TypeError as exc:
                raise ConfigError(f"provider {name}: {exc}") from None
            if providers[name].api_format not in API_FORMATS:
                raise ConfigError(f"provider {name}: unknown api_format {providers[name].api_format!r}")

        models: dict[str, ModelSpec] = {}
        for entry in data.get("models", []):
            try:
                spec = ModelSpec(**entry)
            except TypeError as exc:
                raise ConfigError(f"model {entry.get('name', '?')}: {exc}") from None
            if spec.location not in LOCATIONS:
                raise ConfigError(f"model {spec.name}: unknown location {spec.location!r}")
            if spec.role not in ROLES:
                raise ConfigError(f"model {spec.name}: unknown role {spec.role!r}")
            if spec.name in models:
                raise ConfigError(f"duplicate model {spec.name}")
            if spec.provider:
                prov = providers.get(spec.provider)
                if prov is None:
                    raise ConfigError(f"model {spec.name}: unknown provider {spec.provider!r}")
                spec.base_url = spec.base_url or prov.base_url
                spec.api_format = spec.api_format or prov.api_format
                spec.api_key_env = spec.api_key_env or prov.api_key_env
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
            embodiment_families=data.get("embodiment_families", {}),
            pipeline=data.get("pipeline", []),
            providers=providers,
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
