"""Tests for AdaptorConfig resolution, precedence, and validation."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest

from slopon_harbor.config import (
    DEFAULT_NODE_VERSION,
    REASONING_EFFORT_VALUES,
    AdaptorConfig,
    AdaptorConfigError,
)


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Redirect ~/.slopon/config.json to a temp file; return its path."""
    slopon_dir = tmp_path / ".slopon"
    slopon_dir.mkdir()
    config_path = slopon_dir / "config.json"
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    # expanduser() prefers USERPROFILE on Windows.
    monkeypatch.setenv("USERPROFILE", home)
    for var in (
        "SLOPON_BACKEND_URL",
        "SLOPON_BACKEND_API_KEY",
        "SLOPON_BACKEND_PUBLIC_URL",
        "SLOPON_LLM_API_KEY",
        "SLOPON_LLM_CONTEXT_SIZE",
        "SLOPON_LLM_SUPPORTS_IMAGE",
        "SLOPON_LLM_TEMPERATURE",
        "SLOPON_LLM_REASONING_EFFORT",
        "SLOPON_RUNNER_ONLINE_TIMEOUT_SEC",
    ):
        monkeypatch.delenv(var, raising=False)
    return config_path


@pytest.fixture
def proc_llm_key(monkeypatch):
    monkeypatch.setenv("SLOPON_LLM_API_KEY", "proc-llm-key")


@pytest.fixture
def runtime_dir(tmp_path):
    runtime = tmp_path / "slopon-runtime"
    runtime.mkdir()
    for name in ("runner.js", "package.json"):
        (runtime / name).write_text("stub", encoding="utf-8")
    (runtime / "prompts").mkdir()
    (runtime / "prompts" / "system.md").write_text("prompt", encoding="utf-8")
    (runtime / "node_modules").mkdir()
    (runtime / "node_modules" / "ws").mkdir()
    # Decoys that must NOT be required:
    (runtime / "index.js").write_text("server", encoding="utf-8")
    (runtime / "migrations").mkdir()
    return runtime


def write_config(path: Path, **server) -> None:
    path.write_text(json.dumps({"server": server}), encoding="utf-8")


def valid_config(path: Path) -> None:
    write_config(path, port=4200, listenIp="0.0.0.0", apiKey="k" * 40)


def base_env(runtime: Path, **overrides) -> dict[str, str]:
    env = {
        "SLOPON_RUNNER_RUNTIME": str(runtime),
        "SLOPON_LLM_BASE_URL": "https://llm.example.com/v1",
        "SLOPON_LLM_MODEL_ID": "test-model",
        "SLOPON_LLM_CONTEXT_SIZE": "128000",
    }
    env.update(overrides)
    return env


class TestBackendEndpoint:
    def test_derived_url_requires_non_loopback_listen_ip(self, config_home):
        write_config(config_home, port=4200, listenIp="127.0.0.1", apiKey="k" * 40)
        with pytest.raises(AdaptorConfigError, match="listenIp"):
            AdaptorConfig.from_env(base_env(Path("/nonexistent")))

    def test_derived_url_requires_listen_ip_present(self, config_home):
        write_config(config_home, port=4200, apiKey="k" * 40)
        with pytest.raises(AdaptorConfigError, match="listenIp"):
            AdaptorConfig.from_env(base_env(Path("/nonexistent")))

    def test_derived_url_from_non_loopback_listen_ip(
        self, config_home, runtime_dir, proc_llm_key
    ):
        valid_config(config_home)
        cfg = AdaptorConfig.from_env(base_env(runtime_dir))
        assert cfg.backend_url == "ws://0.0.0.0:4200"
        assert cfg.backend_port == 4200
        assert cfg.backend_api_key == "k" * 40

    def test_explicit_backend_url_opts_out_of_listen_ip_check(
        self, config_home, monkeypatch, runtime_dir, proc_llm_key
    ):
        write_config(config_home, port=4200, listenIp="127.0.0.1", apiKey="k" * 40)
        monkeypatch.setenv("SLOPON_BACKEND_URL", "ws://10.1.2.3:9999")
        cfg = AdaptorConfig.from_env(base_env(runtime_dir))
        assert cfg.backend_url == "ws://10.1.2.3:9999"
        assert cfg.backend_port == 9999

    def test_explicit_url_requires_ws_scheme_and_port(
        self, config_home, monkeypatch, runtime_dir, proc_llm_key
    ):
        valid_config(config_home)
        monkeypatch.setenv("SLOPON_BACKEND_URL", "http://10.1.2.3:9999")
        with pytest.raises(AdaptorConfigError, match="ws://"):
            AdaptorConfig.from_env(base_env(runtime_dir))
        monkeypatch.setenv("SLOPON_BACKEND_URL", "ws://10.1.2.3")
        with pytest.raises(AdaptorConfigError, match="explicit port"):
            AdaptorConfig.from_env(base_env(runtime_dir))

    def test_missing_config_json_without_explicit_url(self, config_home):
        with pytest.raises(AdaptorConfigError, match="config"):
            AdaptorConfig.from_env(base_env(Path("/nonexistent")))

    def test_malformed_config_json(self, config_home):
        config_home.write_text("{not json", encoding="utf-8")
        with pytest.raises(AdaptorConfigError, match="not valid JSON"):
            AdaptorConfig.from_env(base_env(Path("/nonexistent")))

    def test_config_json_missing_server_key(self, config_home):
        config_home.write_text(json.dumps({"nope": 1}), encoding="utf-8")
        with pytest.raises(AdaptorConfigError, match="server"):
            AdaptorConfig.from_env(base_env(Path("/nonexistent")))

    def test_config_json_bad_port(self, config_home):
        write_config(config_home, port="4200")
        with pytest.raises(AdaptorConfigError, match="port"):
            AdaptorConfig.from_env(base_env(Path("/nonexistent")))

    def test_public_url_validated(
        self, config_home, monkeypatch, runtime_dir, proc_llm_key
    ):
        valid_config(config_home)
        monkeypatch.setenv("SLOPON_BACKEND_PUBLIC_URL", "ftp://x:1")
        with pytest.raises(AdaptorConfigError, match="SLOPON_BACKEND_PUBLIC_URL"):
            AdaptorConfig.from_env(base_env(runtime_dir))


class TestRequiredAndDefaults:
    def test_missing_llm_base_url(
        self, config_home, runtime_dir, proc_llm_key
    ):
        valid_config(config_home)
        env = base_env(runtime_dir)
        del env["SLOPON_LLM_BASE_URL"]
        with pytest.raises(AdaptorConfigError, match="SLOPON_LLM_BASE_URL"):
            AdaptorConfig.from_env(env)

    def test_missing_llm_api_key(self, config_home, runtime_dir):
        valid_config(config_home)
        with pytest.raises(AdaptorConfigError, match="SLOPON_LLM_API_KEY"):
            AdaptorConfig.from_env(base_env(runtime_dir))

    def test_missing_runner_runtime(self, config_home, proc_llm_key):
        valid_config(config_home)
        with pytest.raises(AdaptorConfigError, match="SLOPON_RUNNER_RUNTIME"):
            AdaptorConfig.from_env(base_env(Path("/nonexistent")))

    def test_runtime_dir_missing_node_modules(
        self, config_home, runtime_dir, proc_llm_key
    ):
        valid_config(config_home)
        shutil.rmtree(runtime_dir / "node_modules")
        with pytest.raises(AdaptorConfigError, match="node_modules"):
            AdaptorConfig.from_env(base_env(runtime_dir))

    def test_missing_model_id_without_model_name(
        self, config_home, proc_llm_key
    ):
        valid_config(config_home)
        env = base_env(Path("/nonexistent"))
        del env["SLOPON_LLM_MODEL_ID"]
        with pytest.raises(AdaptorConfigError, match="SLOPON_LLM_MODEL_ID"):
            AdaptorConfig.from_env(env)

    def test_model_name_fallback(self, config_home, runtime_dir, proc_llm_key):
        valid_config(config_home)
        env = base_env(runtime_dir)
        del env["SLOPON_LLM_MODEL_ID"]
        cfg = AdaptorConfig.from_env(env, model_name="gpt-test")
        assert cfg.llm_model_id == "gpt-test"

    def test_defaults(self, config_home, runtime_dir, proc_llm_key):
        valid_config(config_home)
        cfg = AdaptorConfig.from_env(base_env(runtime_dir))
        assert cfg.container_workdir == "/app"
        assert cfg.llm_type == "openai_compatible"
        assert cfg.llm_context_size == 128000
        assert cfg.runner_online_timeout_sec == 90.0
        assert cfg.node_version == f"v{DEFAULT_NODE_VERSION}"
        assert cfg.backend_public_url is None
        assert cfg.llm_supports_image is None
        assert cfg.llm_temperature is None
        assert cfg.llm_reasoning_effort is None

    def test_node_version_not_major_22_rejected(
        self, config_home, runtime_dir, proc_llm_key
    ):
        valid_config(config_home)
        env = base_env(runtime_dir, SLOPON_NODE_VERSION="24.1.0")
        with pytest.raises(AdaptorConfigError, match="v22"):
            AdaptorConfig.from_env(env)


class TestContextSize:
    def test_missing_context_size(self, config_home, runtime_dir, proc_llm_key):
        valid_config(config_home)
        env = base_env(runtime_dir)
        del env["SLOPON_LLM_CONTEXT_SIZE"]
        with pytest.raises(AdaptorConfigError, match="SLOPON_LLM_CONTEXT_SIZE"):
            AdaptorConfig.from_env(env)

    def test_non_numeric_context_size(
        self, config_home, runtime_dir, proc_llm_key
    ):
        valid_config(config_home)
        env = base_env(runtime_dir, SLOPON_LLM_CONTEXT_SIZE="big")
        with pytest.raises(AdaptorConfigError, match="positive integer"):
            AdaptorConfig.from_env(env)

    @pytest.mark.parametrize("raw", ["0", "-5"])
    def test_non_positive_context_size(
        self, config_home, runtime_dir, proc_llm_key, raw
    ):
        valid_config(config_home)
        env = base_env(runtime_dir, SLOPON_LLM_CONTEXT_SIZE=raw)
        with pytest.raises(AdaptorConfigError, match="positive integer"):
            AdaptorConfig.from_env(env)

    def test_valid_context_size_lands_on_config(
        self, config_home, runtime_dir, proc_llm_key
    ):
        valid_config(config_home)
        cfg = AdaptorConfig.from_env(
            base_env(runtime_dir, SLOPON_LLM_CONTEXT_SIZE="200000")
        )
        assert cfg.llm_context_size == 200000

    def test_context_size_agent_env_beats_process_env(
        self, config_home, monkeypatch, runtime_dir, proc_llm_key
    ):
        valid_config(config_home)
        monkeypatch.setenv("SLOPON_LLM_CONTEXT_SIZE", "111")
        cfg = AdaptorConfig.from_env(
            base_env(runtime_dir, SLOPON_LLM_CONTEXT_SIZE="222")
        )
        assert cfg.llm_context_size == 222


class TestOptionalLlmSettings:
    def test_unset_means_none(self, config_home, runtime_dir, proc_llm_key):
        valid_config(config_home)
        cfg = AdaptorConfig.from_env(base_env(runtime_dir))
        assert cfg.llm_supports_image is None
        assert cfg.llm_temperature is None
        assert cfg.llm_reasoning_effort is None

    @pytest.mark.parametrize("raw,expected", [("true", True), ("false", False)])
    def test_supports_image_valid(
        self, config_home, monkeypatch, runtime_dir, proc_llm_key, raw, expected
    ):
        valid_config(config_home)
        monkeypatch.setenv("SLOPON_LLM_SUPPORTS_IMAGE", raw)
        cfg = AdaptorConfig.from_env(base_env(runtime_dir))
        assert cfg.llm_supports_image is expected

    @pytest.mark.parametrize("raw", ["yes", "1", "True"])
    def test_supports_image_invalid(
        self, config_home, monkeypatch, runtime_dir, proc_llm_key, raw
    ):
        valid_config(config_home)
        monkeypatch.setenv("SLOPON_LLM_SUPPORTS_IMAGE", raw)
        with pytest.raises(AdaptorConfigError, match="SLOPON_LLM_SUPPORTS_IMAGE"):
            AdaptorConfig.from_env(base_env(runtime_dir))

    @pytest.mark.parametrize("raw,expected", [("0.7", 0.7), ("0", 0.0)])
    def test_temperature_valid(
        self, config_home, monkeypatch, runtime_dir, proc_llm_key, raw, expected
    ):
        valid_config(config_home)
        monkeypatch.setenv("SLOPON_LLM_TEMPERATURE", raw)
        cfg = AdaptorConfig.from_env(base_env(runtime_dir))
        assert cfg.llm_temperature == expected

    def test_temperature_not_a_number(
        self, config_home, monkeypatch, runtime_dir, proc_llm_key
    ):
        valid_config(config_home)
        monkeypatch.setenv("SLOPON_LLM_TEMPERATURE", "abc")
        with pytest.raises(AdaptorConfigError, match="SLOPON_LLM_TEMPERATURE"):
            AdaptorConfig.from_env(base_env(runtime_dir))

    @pytest.mark.parametrize("raw", ["nan", "inf", "-inf"])
    def test_temperature_must_be_finite(
        self, config_home, monkeypatch, runtime_dir, proc_llm_key, raw
    ):
        valid_config(config_home)
        monkeypatch.setenv("SLOPON_LLM_TEMPERATURE", raw)
        with pytest.raises(AdaptorConfigError, match="finite"):
            AdaptorConfig.from_env(base_env(runtime_dir))

    @pytest.mark.parametrize("raw", REASONING_EFFORT_VALUES)
    def test_reasoning_effort_valid(
        self, config_home, monkeypatch, runtime_dir, proc_llm_key, raw
    ):
        valid_config(config_home)
        monkeypatch.setenv("SLOPON_LLM_REASONING_EFFORT", raw)
        cfg = AdaptorConfig.from_env(base_env(runtime_dir))
        assert cfg.llm_reasoning_effort == raw

    @pytest.mark.parametrize("raw", ["High", "extreme"])
    def test_reasoning_effort_invalid(
        self, config_home, monkeypatch, runtime_dir, proc_llm_key, raw
    ):
        valid_config(config_home)
        monkeypatch.setenv("SLOPON_LLM_REASONING_EFFORT", raw)
        with pytest.raises(AdaptorConfigError, match="SLOPON_LLM_REASONING_EFFORT"):
            AdaptorConfig.from_env(base_env(runtime_dir))

    @pytest.mark.parametrize(
        "var",
        [
            "SLOPON_LLM_SUPPORTS_IMAGE",
            "SLOPON_LLM_TEMPERATURE",
            "SLOPON_LLM_REASONING_EFFORT",
        ],
    )
    def test_empty_string_is_unset(
        self, config_home, monkeypatch, runtime_dir, proc_llm_key, var
    ):
        valid_config(config_home)
        monkeypatch.setenv(var, "")
        cfg = AdaptorConfig.from_env(base_env(runtime_dir))
        assert cfg.llm_supports_image is None
        assert cfg.llm_temperature is None
        assert cfg.llm_reasoning_effort is None

    def test_agent_env_beats_process_env(
        self, config_home, monkeypatch, runtime_dir, proc_llm_key
    ):
        valid_config(config_home)
        monkeypatch.setenv("SLOPON_LLM_SUPPORTS_IMAGE", "false")
        monkeypatch.setenv("SLOPON_LLM_TEMPERATURE", "0.1")
        monkeypatch.setenv("SLOPON_LLM_REASONING_EFFORT", "low")
        cfg = AdaptorConfig.from_env(
            base_env(
                runtime_dir,
                SLOPON_LLM_SUPPORTS_IMAGE="true",
                SLOPON_LLM_TEMPERATURE="0.9",
                SLOPON_LLM_REASONING_EFFORT="high",
            )
        )
        assert cfg.llm_supports_image is True
        assert cfg.llm_temperature == 0.9
        assert cfg.llm_reasoning_effort == "high"


class TestPrecedenceAndSecrets:
    def test_agent_env_beats_process_env(
        self, config_home, monkeypatch, runtime_dir, proc_llm_key
    ):
        valid_config(config_home)
        monkeypatch.setenv("SLOPON_LLM_BASE_URL", "https://proc.example.com/v1")
        monkeypatch.setenv("SLOPON_CONTAINER_WORKDIR", "/procwork")
        cfg = AdaptorConfig.from_env(
            base_env(
                runtime_dir,
                SLOPON_LLM_BASE_URL="https://agent.example.com/v1",
                SLOPON_CONTAINER_WORKDIR="/agentwork",
            )
        )
        assert cfg.llm_base_url == "https://agent.example.com/v1"
        assert cfg.container_workdir == "/agentwork"

    def test_process_env_beats_defaults(
        self, config_home, monkeypatch, runtime_dir, proc_llm_key
    ):
        valid_config(config_home)
        monkeypatch.setenv("SLOPON_RUNNER_ONLINE_TIMEOUT_SEC", "5")
        cfg = AdaptorConfig.from_env(base_env(runtime_dir))
        assert cfg.runner_online_timeout_sec == 5.0

    def test_backend_api_key_refused_from_agent_env(
        self, config_home, runtime_dir, proc_llm_key, caplog
    ):
        valid_config(config_home)
        with caplog.at_level(logging.WARNING):
            cfg = AdaptorConfig.from_env(
                base_env(runtime_dir, SLOPON_BACKEND_API_KEY="leaked-key")
            )
        assert cfg.backend_api_key == "k" * 40
        assert any(
            "SLOPON_BACKEND_API_KEY" in rec.message for rec in caplog.records
        )

    def test_llm_api_key_refused_from_agent_env(
        self, config_home, runtime_dir, proc_llm_key, caplog
    ):
        valid_config(config_home)
        with caplog.at_level(logging.WARNING):
            cfg = AdaptorConfig.from_env(
                base_env(runtime_dir, SLOPON_LLM_API_KEY="leaked-key")
            )
        assert cfg.llm_api_key == "proc-llm-key"
        assert any(
            "SLOPON_LLM_API_KEY" in rec.message for rec in caplog.records
        )

    def test_llm_api_key_from_process_env_ok(
        self, config_home, runtime_dir, proc_llm_key
    ):
        valid_config(config_home)
        cfg = AdaptorConfig.from_env(base_env(runtime_dir))
        assert cfg.llm_api_key == "proc-llm-key"
