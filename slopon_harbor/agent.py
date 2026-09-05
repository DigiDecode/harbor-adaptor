"""SlopOnAgent — a harbor agent backed by the SlopOn backend's agentic loop.

Lifecycle:

- ``setup()`` provisions the backend side (provider with a ``contextSize``
  compaction-gate assertion, compaction enable-assert, runner row), then
  the in-container side (Node, runtime tarball upload + unpack, gateway
  discovery, runner start), waits for the runner to come online and
  finishes backend provisioning (project, source folder, bot, approval
  overrides). The setup-phase client connection is closed at the end.
- ``run()`` opens a fresh client connection per call (harbor invokes
  ``run()`` once per step on multi-step tasks and ``BaseAgent`` has no
  teardown hook), pre-creates the chat explicitly so a cancelled stream
  can still be stopped and read back, and streams with no client deadline
  (harbor's ``asyncio.wait_for`` owns cancellation). Context compaction
  is always on: when the backend swaps the chat mid-stream, ``run()``
  re-streams the successor chat (no ``message`` — the backend resumes
  from persisted history) until a stream ends with no successor, then
  records the summed token usage of every chat in the chain into the
  context and writes ``history.json`` (final chat) plus
  ``history-chain.json`` (all chats, multi-chat runs only) next to the
  agent logs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import re
import tempfile
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from slopon_harbor import __version__
from slopon_harbor.backend_client import (
    BackendClientError,
    BackendRpcError,
    PushCallback,
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
CHAIN_HISTORY_FILENAME = "history-chain.json"
ERROR_FILENAME = "error.json"
POST_CANCEL_CALL_TIMEOUT_SEC = 10.0
METADATA_CHATS_KEY = "slopon_chats"
COMPACTION_FAILED_CODE = "COMPACTION_FAILED"

_IPV4_RE = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}")


class _CompactionWatch:
    """``on_push`` sink for the compaction lifecycle pushes of one ``run()``.

    Invoked synchronously on the client's reader loop, so it only logs and
    records — never blocks. ``chat.compactionFailed`` is the only failure
    signal a failed compaction produces (no DB trace is left behind), so
    the recorded entry is checked after every stream end. Only
    ``started``/``failed`` target the streaming connection;
    ``completed`` is a UI-only broadcast (``broadcastToUiClients`` gates
    on ``isUi``) a non-UI adaptor never receives — the run's own
    per-hop successor log is the surviving completion observability.
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self.failures: dict[int, str] = {}

    def __call__(self, method: str, params: dict[str, Any]) -> None:
        if method == "chat.compactionStarted":
            self._logger.info("compaction started for chat %s", params.get("chatId"))
        elif method == "chat.compactionFailed":
            raw_id = params.get("chatId")
            error = str(params.get("error") or "unknown error")
            try:
                chat_id = int(raw_id)
            except (TypeError, ValueError):
                self._logger.warning(
                    "compaction failed push has unusable chatId %r: %s",
                    raw_id,
                    error,
                )
                return
            self.failures[chat_id] = error
            self._logger.warning("compaction failed for chat %s: %s", chat_id, error)


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

    def _connect(self, on_push: PushCallback | None = None) -> SlopOnBackendClient:
        client = SlopOnBackendClient(
            self._config.backend_url,
            self._config.backend_api_key,
            logger=self.logger,
            on_push=on_push,
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

            # Steps 1-3: provider, compaction enable-assert, runner row + token.
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

        watch = _CompactionWatch(self.logger)
        client = self._connect(on_push=watch)
        try:
            await client.connect()
            chat = await client.call("chat.create", {"botId": bot_id})
            first_chat_id = int(chat["id"])
            chain = [first_chat_id]
            current = first_chat_id

            def on_chunk(data) -> None:
                self.logger.debug("chat chunk: %s", data)

            input_tokens = cache_tokens = output_tokens = 0
            breakdown: list[dict[str, Any]] = []
            try:
                while True:
                    params: dict[str, Any] = {"chatId": current, "botId": bot_id}
                    if current == first_chat_id:
                        # Only the first stream carries the instruction; a
                        # compacted successor resumes from persisted history.
                        params["message"] = instruction
                    await client.stream(
                        "chat.stream", params, on_chunk=on_chunk, timeout=None
                    )
                    # Each chat's usage is queried exactly once, as soon as
                    # its stream ends; the failure path below exits before
                    # any accounting matters.
                    usage = await client.call(
                        "chat.getTokenUsage", {"chatId": current}
                    )
                    input_tokens += usage.get("inputTokens") or 0
                    cache_tokens += usage.get("cacheHitTokens") or 0
                    output_tokens += usage.get("outputTokens") or 0
                    breakdown.append(
                        {
                            "chatId": current,
                            "inputTokens": usage.get("inputTokens"),
                            "cacheHitTokens": usage.get("cacheHitTokens"),
                            "outputTokens": usage.get("outputTokens"),
                        }
                    )
                    failure = watch.failures.get(current)
                    if failure is not None:
                        self._write_json(
                            ERROR_FILENAME,
                            {"code": COMPACTION_FAILED_CODE, "message": failure},
                        )
                        raise RuntimeError(
                            f"backend compaction failed for chat {current}: "
                            f"{failure}"
                        )
                    successor = await self._find_successor(client, current)
                    if successor is None:
                        break
                    self.logger.info(
                        "chat %s was compacted; continuing in chat %s",
                        current,
                        successor,
                    )
                    chain.append(successor)
                    current = successor

                context.n_input_tokens = input_tokens
                context.n_cache_tokens = cache_tokens
                context.n_output_tokens = output_tokens
                context.metadata = {
                    **(context.metadata or {}),
                    METADATA_CHATS_KEY: breakdown,
                }

                final_history = await client.call(
                    "chat.getHistory", {"chatId": current}
                )
                self._write_json(HISTORY_FILENAME, final_history)
                if len(chain) > 1:
                    entries = [
                        await client.call("chat.getHistory", {"chatId": chat_id})
                        for chat_id in chain[:-1]
                    ]
                    entries.append(final_history)
                    self._write_json(CHAIN_HISTORY_FILENAME, entries)
            except asyncio.CancelledError:
                # Harbor's agent-phase timeout/abort: stop the stream
                # best-effort, salvage the partial transcript, re-raise.
                await self._salvage_after_cancel(client, current, chain)
                raise
            except BackendRpcError as err:
                self._write_json(
                    ERROR_FILENAME, {"code": err.code, "message": err.message}
                )
                raise
        finally:
            await client.close()

    async def _find_successor(
        self, client: SlopOnBackendClient, chat_id: int
    ) -> int | None:
        """Return the compaction successor of ``chat_id``, or ``None``.

        ``chat.get`` resolves ``continuationChatId`` server-side — the
        latest chat with ``previousChatId = chat_id`` — in the same
        single, race-free call (no pagination surface).
        """
        chat = await client.call("chat.get", {"chatId": chat_id})
        if "continuationChatId" not in chat:
            # A pre-372 backend omits the key entirely. Reading absence
            # as "no successor" would silently truncate a compacted run.
            raise RuntimeError(
                f"chat.get for chat {chat_id} returned no continuationChatId "
                "field; the backend predates task-372 — point "
                "SLOPON_RUNNER_RUNTIME at a current release"
            )
        raw = chat["continuationChatId"]
        if raw is None:
            return None
        successor = int(raw)
        if successor == chat_id:
            # Defensive: unreachable while compaction always creates a
            # new row, but guards against backend contract drift.
            raise RuntimeError(
                f"chat.get returned chat {chat_id} as its own continuation; "
                "refusing to loop forever"
            )
        return successor

    async def _salvage_after_cancel(
        self, client: SlopOnBackendClient, chat_id: int, chain: list[int]
    ) -> None:
        """Best-effort stop + partial-transcript salvage after cancellation.

        ``chat_id`` is the most recent chat at cancel time; artifacts
        follow the completed-run layout (final chat plus the chain file
        for multi-chat runs), each call individually bounded by
        ``POST_CANCEL_CALL_TIMEOUT_SEC``.
        """
        try:
            await asyncio.wait_for(
                client.call("chat.stopStream", {"chatId": chat_id}),
                timeout=POST_CANCEL_CALL_TIMEOUT_SEC,
            )
        except (BackendClientError, TimeoutError):
            self.logger.warning("chat.stopStream best-effort call failed")
        final_history: dict[str, Any] | None = None
        try:
            final_history = await asyncio.wait_for(
                client.call("chat.getHistory", {"chatId": chat_id}),
                timeout=POST_CANCEL_CALL_TIMEOUT_SEC,
            )
            self._write_json(HISTORY_FILENAME, final_history)
        except (BackendClientError, TimeoutError):
            self.logger.warning(
                "could not fetch partial history for chat %s", chat_id
            )
        if len(chain) <= 1:
            return
        entries: list[dict[str, Any]] = []
        for earlier_id in chain[:-1]:
            try:
                entries.append(
                    await asyncio.wait_for(
                        client.call("chat.getHistory", {"chatId": earlier_id}),
                        timeout=POST_CANCEL_CALL_TIMEOUT_SEC,
                    )
                )
            except (BackendClientError, TimeoutError):
                self.logger.warning(
                    "could not fetch partial history for chat %s", earlier_id
                )
                return
        if final_history is not None:
            entries.append(final_history)
            self._write_json(CHAIN_HISTORY_FILENAME, entries)

    def _write_json(self, filename: str, payload) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        target = self.logs_dir / filename
        target.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        self.logger.info("wrote %s", target)
