"""Hermetic guard: the shipped example job config stays runnable."""

from pathlib import Path

import yaml

EXAMPLE_PATH = (
    Path(__file__).resolve().parent.parent / "examples" / "hello-world-slopon.yaml"
)


def test_example_job_sets_required_agent_env():
    spec = yaml.safe_load(EXAMPLE_PATH.read_text(encoding="utf-8"))
    env = spec["agents"][0]["env"]
    # All three are required at adaptor config load; a missing one fails
    # every job start using the example.
    assert "SLOPON_RUNNER_RUNTIME" in env
    assert "SLOPON_LLM_BASE_URL" in env
    assert "SLOPON_LLM_CONTEXT_SIZE" in env
