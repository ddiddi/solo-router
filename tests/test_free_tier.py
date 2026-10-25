"""The website's free-tier Worker: keys stay server-side and the limits hold."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_free_tier_worker_limits():
    out = json.loads(subprocess.run(["node", str(HERE / "free_tier.mjs")], capture_output=True, text=True,
                                    check=True).stdout)
    assert json.loads(out["status"]["body"]) == {"providers": ["anthropic", "nebius"], "remaining": 1}
    assert out["wrongOrigin"]["status"] == 403
    assert out["selfHosted"]["status"] == 400 and out["noSecret"]["status"] == 400 and out["tooLong"]["status"] == 400

    first = out["first"]
    assert (first["status"], first["format"], first["remaining"]) == (200, "openai", "0")
    assert "ok" in first["body"]
    assert out["second"]["status"] == 429 and "free run" in json.loads(out["second"]["body"])["error"]["message"]
    assert out["otherVisitor"]["status"] == 200 and out["otherVisitor"]["format"] == "anthropic"
    assert out["capped"]["status"] == 429  # DAILY_CAP=2 across all visitors

    nebius, anthropic = out["upstream"]
    assert nebius["url"].startswith("https://api.tokenfactory.nebius.com/v1/chat/completions")
    assert nebius["headers"]["authorization"] == "Bearer nb-secret"
    assert nebius["body"]["model"] == "openai/gpt-oss-120b" and nebius["body"]["max_completion_tokens"] == 100
    assert anthropic["headers"]["x-api-key"] == "ak-secret" and anthropic["body"]["max_tokens"] == 100
    # Visitors are stored as hashes, never raw addresses.
    assert not any("1.1.1.1" in k or "2.2.2.2" in k for k in out["storedKeys"])
