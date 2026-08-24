"""SlopOnAgent — a harbor agent backed by the SlopOn backend's agentic loop.

Lifecycle:

- ``setup()`` provisions the backend side (provider, compaction off,
  runner row), then the in-container side (Node, runtime tarball upload +
  unpack, gateway discovery, runner start), waits for the runner to come
  online and finishes backend provisioning (project, source folder, bot,
  approval overrides). The setup-phase client connection is closed at the
  end.
- ``run()`` opens a fresh client connection per call (harbor invokes
  ``run()`` once per step on multi-step tasks and ``BaseAgent`` has no
  teardown hook), pre-creates the chat explicitly so a cancelled stream
  can still be stopped and read back, streams with no client deadline
  (harbor's ``asyncio.wait_for`` owns cancellation), records token usage
  into the context, and writes ``history.json`` next to the agent logs.
"""

from __future__ import annotations

import asyncio
import json
import platform
import re
import tempfile
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from slopon_harbor import __version__
from slopon_harbor.backend_client import (
    BackendClientError,
    BackendRpcError,
    SlopOnBackendClient,
)
from slopon_harbor.config import AdaptorConfig
from slopon_harbor.container import (
    RUNNER_LOG_PATH,
    RUNTIME_TAR_DEST,
    build_runtime_tar,
    gateway_discovery_command,
    node_provision_commands,
    runner_start_command,
    runner_start_env,
    runtime_unpack_command,
)
from slopon_harbor.provisioning import TrialProvisioner

HISTORY_FILENAME = "history.json"
ERROR_FILENAME = "error.json"
POST_CANCEL_CALL_TIMEOUT_SEC = 10.0

_IPV4_RE = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}")


class SlopOnAgent(BaseAgent):
    SUPPORTS_WINDOWS = False  # egress sidecar + nftables are Linux-only

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = AdaptorConfig.from_env(self.extra_env, self.model_name)
        self._provisioner: TrialProvisioner | None = None
        self._resources = None

    @staticmethod
    def name() -> str:
        return "slopon"

    def version(self) -> str | None:
        return __version__

    # ── helpers ──────────────────────────────────────────────────────────

    def _connect(self) -> SlopOnBackendClient:
        client = SlopOnBackendClient(
            self._config.backend_url,
            self._config.backend_api_key,
            logger=self.logger,
        )
        return client

    async def _exec(
        self,
        environment: BaseEnvironment,
        command: str,
        *,
        env: dict[str, str] | None = None,
    ) -> str:
        result = await environment.exec(command, env=env)
        if result.return_code != 0:
            raise RuntimeError(
                f"in-container command failed (exit {result.return_code}): "
                f"{command.splitlines()[0][:200]}\n"
                f"stdout: {(result.stdout or '')[-2000:]}\n"
                f"stderr: {(result.stderr or '')[-2000:]}"
            )
        return result.stdout or ""

    # ── setup ────────────────────────────────────────────────────────────

    async def setup(self, environment: BaseEnvironment) -> None:
        client = self._connect()
        try:
            await client.connect()
            self._provisioner = TrialProvisioner(
                client,
                self._config,
                self.session_id or "trial",
                logger=self.logger,
            )

            # Steps 1-3: provider, compaction off, runner row + token.
            runner_id, runner_token = (
                await self._provisioner.create_runner_resources()
            )

            # In-container provisioning.
            arch = platform.machine() or "x86_64"
            node_bin = await self._provision_node(environment, arch)
            await self._deliver_runtime(environment)
            public_url = await self._resolve_backend_public_url(environment)
            await self._start_runner(
                environment, node_bin, public_url, runner_token
            )

            # Step 4: wait for the in-container runner to register.
            async def fetch_log_tail() -> str:
                result = await environment.exec(f"tail -n 50 {RUNNER_LOG_PATH}")
                return result.stdout or ""

            await self._provisioner.wait_runner_online(
                runner_id, fetch_log_tail=fetch_log_tail
            )

            # Steps 5-8: project resources against the online runner.
            self._resources = await self._provisioner.create_project_resources(
                runner_id, runner_token
            )
        finally:
            await client.close()

    async def _provision_node(
        self, environment: BaseEnvironment, arch: str
    ) -> str:
        commands = node_provision_commands(arch, self._config.node_version)
        output = ""
        for command in commands:
            output = await self._exec(environment, command)
        node_bin = output.strip().splitlines()[-1].strip() if output.strip() else ""
        if not node_bin or "/" not in node_bin:
            raise RuntimeError(
                f"node provisioning did not report a node binary path "
                f"(got {node_bin!r})"
            )
        self.logger.info("using node binary %r", node_bin)
        return node_bin

    async def _deliver_runtime(self, environment: BaseEnvironment) -> None:
        with tempfile.TemporaryDirectory(prefix="slopon-runtime-") as tmp:
            tar_path = Path(tmp) / "runtime.tar.gz"
            build_runtime_tar(self._config.runner_runtime, tar_path)
            self.logger.info(
                "uploading runner runtime tarball (%d bytes) into the container",
                tar_path.stat().st_size,
            )
            await environment.upload_file(tar_path, RUNTIME_TAR_DEST)
        await self._exec(environment, runtime_unpack_command())

    async def _resolve_backend_public_url(
        self, environment: BaseEnvironment
    ) -> str:
        if self._config.backend_public_url:
            url = self._config.backend_public_url
            self.logger.info("using configured SLOPON_BACKEND_PUBLIC_URL %s", url)
            return url
        output = await self._exec(environment, gateway_discovery_command())
        gateway = output.strip().splitlines()[-1].strip() if output.strip() else ""
        if not _IPV4_RE.fullmatch(gateway):
            raise RuntimeError(
                "could not discover the container's default-route gateway; "
                "set SLOPON_BACKEND_PUBLIC_URL (e.g. "
                "ws://host.docker.internal:4200 on Docker Desktop)"
            )
        url = f"ws://{gateway}:{self._config.backend_port}"
        # Logged so operators know which address to allowlist for
        # allowlist-policy tasks.
        self.logger.info("container reaches the backend via gateway %s", url)
        return url

    async def _start_runner(
        self,
        environment: BaseEnvironment,
        node_bin: str,
        public_url: str,
        runner_token: str,
    ) -> None:
        await self._exec(
            environment,
            runner_start_command(node_bin),
            env=runner_start_env(public_url, runner_token),
        )

    # ── run ──────────────────────────────────────────────────────────────

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if self._provisioner is None or self._resources is None:
            raise RuntimeError("setup() has not completed; cannot run")
        bot_id = self._resources.bot_id

        client = self._connect()
        try:
            await client.connect()
            chat = await client.call("chat.create", {"botId": bot_id})
            chat_id = int(chat["id"])

            def on_chunk(data) -> None:
                self.logger.debug("chat chunk: %s", data)

            try:
                await client.stream(
                    "chat.stream",
                    {"chatId": chat_id, "botId": bot_id, "message": instruction},
                    on_chunk=on_chunk,
                    timeout=None,
                )
            except asyncio.CancelledError:
                # Harbor's agent-phase timeout/abort: stop the stream
                # best-effort, salvage the partial transcript, re-raise.
                await self._salvage_after_cancel(client, chat_id)
                raise
            except BackendRpcError as err:
                self._write_json(
                    ERROR_FILENAME, {"code": err.code, "message": err.message}
                )
                raise

            usage = await client.call("chat.getTokenUsage", {"chatId": chat_id})
            context.n_input_tokens = usage.get("inputTokens")
            context.n_cache_tokens = usage.get("cacheHitTokens")
            context.n_output_tokens = usage.get("outputTokens")

            history = await client.call("chat.getHistory", {"chatId": chat_id})
            self._write_json(HISTORY_FILENAME, history)
        finally:
            await client.close()

    async def _salvage_after_cancel(
        self, client: SlopOnBackendClient, chat_id: int
    ) -> None:
        try:
            await asyncio.wait_for(
                client.call("chat.stopStream", {"chatId": chat_id}),
                timeout=POST_CANCEL_CALL_TIMEOUT_SEC,
            )
        except (BackendClientError, TimeoutError):
            self.logger.warning("chat.stopStream best-effort call failed")
        try:
            history = await asyncio.wait_for(
                client.call("chat.getHistory", {"chatId": chat_id}),
                timeout=POST_CANCEL_CALL_TIMEOUT_SEC,
            )
            self._write_json(HISTORY_FILENAME, history)
        except (BackendClientError, TimeoutError):
            self.logger.warning(
                "could not fetch partial history for chat %s", chat_id
            )

    def _write_json(self, filename: str, payload) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        target = self.logs_dir / filename
        target.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        self.logger.info("wrote %s", target)
