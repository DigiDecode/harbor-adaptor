"""Adaptor configuration: environment + ``~/.slopon/config.json`` resolution.

Precedence (highest first) per variable:

1. agent ``extra_env`` (harbor ``agent.env`` / ``--ae``),
2. the harbor process environment,
3. defaults derived from ``~/.slopon/config.json``.

Secrets are special-cased: ``SLOPON_BACKEND_API_KEY`` and
``SLOPON_LLM_API_KEY`` are refused from agent env. Harbor injects agent
env into every container exec (setup and agent phase), which would leak
the keys into the agent-under-test's sandbox, and the in-container runner
receives LLM credentials inside the job spec anyway — so in-container
presence buys nothing and only adds exposure.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from slopon_harbor.container import validate_pinned_node_version

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_JSON = Path("~/.slopon/config.json")
DEFAULT_CONTAINER_WORKDIR = "/app"
DEFAULT_LLM_TYPE = "openai_compatible"
DEFAULT_RUNNER_ONLINE_TIMEOUT_SEC = 90.0
DEFAULT_NODE_VERSION = "22.17.1"

# Mirrors the backend api-provider schema's reasoningEffortValues enum.
REASONING_EFFORT_VALUES = ("none", "minimal", "low", "medium", "high", "xhigh")

AGENT_ENV_FORBIDDEN = (
    "SLOPON_BACKEND_API_KEY",
    "SLOPON_LLM_API_KEY",
)

_RUNTIME_REQUIRED_ENTRIES = ("runner.js", "prompts", "node_modules")


class AdaptorConfigError(Exception):
    """Raised when adaptor configuration is missing or invalid."""


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "::1", "localhost", "0:0:0:0:0:0:0:1")


def _require_ws_port(url: str, *, what: str) -> int:
    parsed = urlparse(url)
    if parsed.scheme not in ("ws", "wss"):
        raise AdaptorConfigError(
            f"{what} must be a ws:// or wss:// URL, got: {url!r}"
        )
    if parsed.port is None:
        raise AdaptorConfigError(
            f"{what} must carry an explicit port (the container-side backend "
            f"URL is derived from it), got: {url!r}"
        )
    return parsed.port


@dataclass(frozen=True)
class _BootstrapConfig:
    port: int | None
    listen_ip: str | None
    api_key: str | None

    @classmethod
    def load(cls, path: Path) -> _BootstrapConfig:
        """Parse ``~/.slopon/config.json`` per the backend bootstrap schema."""
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError as err:
            raise AdaptorConfigError(
                f"backend config not found at {path} — pre-provision the "
                "benchmark backend (see the adaptor README) or set "
                "SLOPON_BACKEND_URL / SLOPON_BACKEND_API_KEY explicitly"
            ) from err
        except OSError as err:
            raise AdaptorConfigError(f"cannot read backend config {path}: {err}") from err
        try:
            parsed = json.loads(raw)
        except ValueError as err:
            raise AdaptorConfigError(
                f"backend config {path} is not valid JSON: {err}"
            ) from err
        if not isinstance(parsed, dict) or not isinstance(parsed.get("server"), dict):
            raise AdaptorConfigError(
                f"backend config {path} must contain a 'server' object with "
                "'port' and optionally 'listenIp'/'apiKey'"
            )
        server = parsed["server"]
        port = server.get("port")
        if port is not None and (not isinstance(port, int) or port <= 0):
            raise AdaptorConfigError(
                f"backend config {path}: server.port must be a positive integer"
            )
        listen_ip = server.get("listenIp")
        if listen_ip is not None and not isinstance(listen_ip, str):
            raise AdaptorConfigError(
                f"backend config {path}: server.listenIp must be a string"
            )
        api_key = server.get("apiKey")
        if api_key is not None and not isinstance(api_key, str):
            raise AdaptorConfigError(
                f"backend config {path}: server.apiKey must be a string"
            )
        return cls(port=port, listen_ip=listen_ip, api_key=api_key)


@dataclass(frozen=True)
class AdaptorConfig:
    """Fully validated adaptor configuration."""

    backend_url: str
    backend_api_key: str
    backend_port: int
    backend_public_url: str | None
    runner_runtime: Path
    container_workdir: str
    llm_type: str
    llm_base_url: str
    llm_api_key: str
    llm_model_id: str
    llm_context_size: int
    llm_supports_image: bool | None
    llm_temperature: float | None
    llm_reasoning_effort: str | None
    runner_online_timeout_sec: float
    node_version: str

    @classmethod
    def from_env(
        cls,
        agent_env: dict[str, str] | None = None,
        model_name: str | None = None,
    ) -> AdaptorConfig:
        agent_env = dict(agent_env or {})
        for forbidden in AGENT_ENV_FORBIDDEN:
            if forbidden in agent_env:
                logger.warning(
                    "%s was set via agent env (agent.env / --ae); refusing it. "
                    "Harbor injects agent env into every container exec, which "
                    "would leak the key into the task container. See the "
                    "adaptor README for supported secret placement.",
                    forbidden,
                )
                agent_env.pop(forbidden)

        def resolve(name: str) -> str | None:
            if name in agent_env:
                return agent_env[name]
            return os.environ.get(name)

        # ── backend endpoint + credentials ────────────────────────────────
        explicit_url = resolve("SLOPON_BACKEND_URL")
        bootstrap = _BootstrapConfig.load(DEFAULT_CONFIG_JSON.expanduser())

        if explicit_url is not None:
            backend_port = _require_ws_port(explicit_url, what="SLOPON_BACKEND_URL")
            backend_url = explicit_url
        else:
            if bootstrap.port is None:
                raise AdaptorConfigError(
                    f"{DEFAULT_CONFIG_JSON} is missing server.port and no "
                    "SLOPON_BACKEND_URL is set — cannot determine the backend "
                    "endpoint"
                )
            listen_ip = bootstrap.listen_ip
            if listen_ip is None or _is_loopback(listen_ip):
                raise AdaptorConfigError(
                    f"{DEFAULT_CONFIG_JSON} server.listenIp is "
                    f"{'missing' if listen_ip is None else f'{listen_ip!r} (loopback)'} "
                    "and no SLOPON_BACKEND_URL is set. The backend defaults to "
                    "127.0.0.1, which task containers can never reach. Set "
                    '"server.listenIp": "0.0.0.0" (or the host LAN IP) in the '
                    "backend config and restart the backend, or pass an "
                    "explicit SLOPON_BACKEND_URL."
                )
            backend_port = bootstrap.port
            backend_url = f"ws://{listen_ip}:{bootstrap.port}"

        api_key = resolve("SLOPON_BACKEND_API_KEY")
        if api_key is None:
            if bootstrap.api_key is None:
                raise AdaptorConfigError(
                    f"no backend API key: {DEFAULT_CONFIG_JSON} has no "
                    "server.apiKey and SLOPON_BACKEND_API_KEY is not set in "
                    "the harbor process environment"
                )
            api_key = bootstrap.api_key

        public_url = resolve("SLOPON_BACKEND_PUBLIC_URL")
        if public_url is not None:
            _require_ws_port(public_url, what="SLOPON_BACKEND_PUBLIC_URL")

        # ── required LLM + runtime settings ───────────────────────────────
        llm_base_url = resolve("SLOPON_LLM_BASE_URL")
        if not llm_base_url:
            raise AdaptorConfigError(
                "SLOPON_LLM_BASE_URL is required (agent env or harbor process "
                "env): the LLM provider base URL, e.g. https://api.example.com/v1"
            )
        llm_api_key = os.environ.get("SLOPON_LLM_API_KEY")
        if not llm_api_key:
            raise AdaptorConfigError(
                "SLOPON_LLM_API_KEY is required in the harbor PROCESS "
                "environment (never agent.env/--ae — harbor injects agent env "
                "into the task container; the runner gets credentials inside "
                "its job spec)"
            )
        llm_model_id = resolve("SLOPON_LLM_MODEL_ID")
        if not llm_model_id:
            if not model_name:
                raise AdaptorConfigError(
                    "SLOPON_LLM_MODEL_ID is required (or pass harbor -m/--model)"
                )
            llm_model_id = model_name

        context_raw = resolve("SLOPON_LLM_CONTEXT_SIZE")
        if not context_raw:
            raise AdaptorConfigError(
                "SLOPON_LLM_CONTEXT_SIZE is required (agent env or harbor process "
                "env): the model context window in tokens. Compaction is always "
                "on; without it the backend compaction stop condition never fires."
            )
        try:
            llm_context_size = int(context_raw)
        except ValueError:
            llm_context_size = -1  # fall through to the range check below
        if llm_context_size <= 0:
            raise AdaptorConfigError(
                f"SLOPON_LLM_CONTEXT_SIZE must be a positive integer (tokens), "
                f"got {context_raw!r}"
            )

        # ── optional LLM settings (None = unset) ─────────────────────────
        supports_raw = resolve("SLOPON_LLM_SUPPORTS_IMAGE")
        if supports_raw:
            if supports_raw not in ("true", "false"):
                raise AdaptorConfigError(
                    "SLOPON_LLM_SUPPORTS_IMAGE must be 'true' or 'false', "
                    f"got {supports_raw!r}"
                )
            llm_supports_image = supports_raw == "true"
        else:
            llm_supports_image = None

        temperature_raw = resolve("SLOPON_LLM_TEMPERATURE")
        if temperature_raw:
            try:
                llm_temperature = float(temperature_raw)
            except ValueError as err:
                raise AdaptorConfigError(
                    f"SLOPON_LLM_TEMPERATURE must be a finite number, "
                    f"got {temperature_raw!r}"
                ) from err
            if not math.isfinite(llm_temperature):
                raise AdaptorConfigError(
                    f"SLOPON_LLM_TEMPERATURE must be finite, "
                    f"got {temperature_raw!r}"
                )
        else:
            llm_temperature = None

        effort_raw = resolve("SLOPON_LLM_REASONING_EFFORT")
        if effort_raw:
            if effort_raw not in REASONING_EFFORT_VALUES:
                raise AdaptorConfigError(
                    "SLOPON_LLM_REASONING_EFFORT must be one of "
                    f"{'|'.join(REASONING_EFFORT_VALUES)}, got {effort_raw!r}"
                )
            llm_reasoning_effort = effort_raw
        else:
            llm_reasoning_effort = None

        runtime_raw = resolve("SLOPON_RUNNER_RUNTIME")
        if not runtime_raw:
            raise AdaptorConfigError(
                "SLOPON_RUNNER_RUNTIME is required: host path to the prepared "
                "runner runtime (the release tree after `npm install`; must "
                "contain runner.js, prompts/ and node_modules/)"
            )
        runner_runtime = Path(runtime_raw).expanduser()
        _validate_runtime_dir(runner_runtime)

        # ── optional settings ─────────────────────────────────────────────
        timeout_raw = resolve("SLOPON_RUNNER_ONLINE_TIMEOUT_SEC")
        try:
            online_timeout = (
                float(timeout_raw) if timeout_raw else DEFAULT_RUNNER_ONLINE_TIMEOUT_SEC
            )
        except ValueError as err:
            raise AdaptorConfigError(
                f"SLOPON_RUNNER_ONLINE_TIMEOUT_SEC must be a number, "
                f"got {timeout_raw!r}"
            ) from err
        if online_timeout <= 0:
            raise AdaptorConfigError(
                "SLOPON_RUNNER_ONLINE_TIMEOUT_SEC must be positive, "
                f"got {online_timeout}"
            )

        try:
            node_version = validate_pinned_node_version(
                resolve("SLOPON_NODE_VERSION") or DEFAULT_NODE_VERSION
            )
        except ValueError as err:
            raise AdaptorConfigError(str(err)) from err

        return cls(
            backend_url=backend_url,
            backend_api_key=api_key,
            backend_port=backend_port,
            backend_public_url=public_url,
            runner_runtime=runner_runtime,
            container_workdir=resolve("SLOPON_CONTAINER_WORKDIR")
            or DEFAULT_CONTAINER_WORKDIR,
            llm_type=resolve("SLOPON_LLM_TYPE") or DEFAULT_LLM_TYPE,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model_id=llm_model_id,
            llm_context_size=llm_context_size,
            llm_supports_image=llm_supports_image,
            llm_temperature=llm_temperature,
            llm_reasoning_effort=llm_reasoning_effort,
            runner_online_timeout_sec=online_timeout,
            node_version=node_version,
        )


def _validate_runtime_dir(runtime_dir: Path) -> None:
    missing: list[str] = []
    if not runtime_dir.is_dir():
        raise AdaptorConfigError(
            f"SLOPON_RUNNER_RUNTIME {runtime_dir} does not exist or is not a "
            "directory. Prepare it from the published release zip (unzip, then "
            "`npm install` inside the backend/ subtree) — see the adaptor README."
        )
    if not (runtime_dir / "runner.js").is_file():
        missing.append("runner.js")
    if not (runtime_dir / "prompts").is_dir():
        missing.append("prompts/")
    if not (runtime_dir / "node_modules").is_dir():
        missing.append("node_modules/")
    if missing:
        raise AdaptorConfigError(
            f"SLOPON_RUNNER_RUNTIME {runtime_dir} is missing {', '.join(missing)}. "
            "It must be the release tree AFTER `npm install` (the published zip "
            "ships no node_modules). See the adaptor README."
        )
