"""Backend-side provisioning: provider, runner, project, bot, permissions.

One ``TrialProvisioner`` per trial drives the RPC flow against an
authenticated client connection:

1. ``ensure_provider`` — reuse the benchmark provider row by name, create
   it on first use (never sending ``reasoningEffort`` — the service
   defaults it to ``'high'`` and an explicit ``null`` breaks
   ``openai_compatible`` providers), then assert via ``apiProvider.update``
   the provider ``contextSize`` compaction gate (rows created before
   compaction was enabled lack it) plus any configured optional LLM
   settings (``supportsImage`` / ``temperature`` / ``reasoningEffort``;
   unset env means the key is omitted so the stored row value survives).
2. ``ensure_compaction_enabled`` — global ``config.set`` asserting
   ``global.context-compaction.enabled = 'true'`` on every setup
   (self-heals ``'false'`` rows written by older adaptor versions;
   compacted chats swap mid-run and the agent continues in the
   successor chat).
3. ``create_runner`` — runner row with a per-run random suffix (duplicate
   names are rejected backend-wide and tokens are unrecoverable).
4. ``wait_runner_online`` — poll ``runner.list`` until the in-container
   runner registers.
5. ``create_project_resources`` — project, runner assignment (never
   ``null`` — that would resolve to the backend's own host-local runner),
   source folder, benchmark bot, and per-tool approval overrides derived
   from ``tool.list``.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from slopon_harbor.backend_client import (
    BackendClientError,
    BackendRpcError,
    SlopOnBackendClient,
)
from slopon_harbor.config import AdaptorConfig

# Base (un-namespaced) tool ids — bot.create persists base ids and the
# backend strips the namespace when matching. Excludes client-forwarded /
# interactive tools (ask_question) and configurable tools (ask_another_llm,
# task management, delegation).
INCLUDED_TOOLS = [
    "list_directory",
    "read_file",
    "search_files",
    "search_content",
    "write_file",
    "edit_file",
    "move_file",
    "delete_file",
    "execute_command",
    "manage_process",
    "lsp_goto_definition",
    "lsp_find_references",
]

BOT_NAME = "benchmark"
PROVIDER_NAME_PREFIX = "benchmark-"
COMPACTION_CONFIG_KEY = "global.context-compaction.enabled"
RUNNER_POLL_INTERVAL_SEC = 2.0
RUNNER_LOG_TAIL_LINES = 50

LogTailProvider = Callable[[], Awaitable[str]] | None


class ProvisioningError(BackendClientError):
    """Raised when the trial resource flow cannot complete."""


@dataclass(frozen=True)
class TrialResources:
    provider_id: int
    runner_id: int
    runner_token: str
    project_id: int
    source_folder_id: int
    bot_id: int


def derive_tool_namespace_prefix(tool_ids: list[str]) -> str:
    """Longest common prefix of ``tool_ids`` that ends in ``_``.

    ``tool.list`` returns namespaced ids (implementation detail of the
    backend — ``noslop_v1_`` today); deriving it from the response keeps
    the adaptor correct if the prefix ever changes.
    """
    if not tool_ids:
        return ""
    prefix = tool_ids[0]
    for tool_id in tool_ids[1:]:
        while not tool_id.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    # Cut back to the last underscore so we never strip into a tool name.
    return prefix[: prefix.rfind("_") + 1] if "_" in prefix else ""


def strip_tool_namespace(prefixed_id: str, prefix: str) -> str:
    if prefix and prefixed_id.startswith(prefix):
        return prefixed_id[len(prefix) :]
    return prefixed_id


class TrialProvisioner:
    """Per-trial backend provisioning (steps 1-8 of the plan)."""

    def __init__(
        self,
        client: SlopOnBackendClient,
        config: AdaptorConfig,
        session_id: str,
        *,
        logger: logging.Logger | None = None,
    ):
        self._client = client
        self._config = config
        self._session_id = session_id or "trial"
        self._logger = logger or logging.getLogger(__name__)
        self._resources: TrialResources | None = None
        self._provider_id: int | None = None
        # One suffix per trial: the runner and the project share it so the
        # names stay symmetric and never collide across same-session reruns.
        self._suffix = secrets.token_hex(2)

    @property
    def resources(self) -> TrialResources:
        if self._resources is None:
            raise ProvisioningError("trial resources not provisioned yet")
        return self._resources

    # ── steps 1-3: before the container runner starts ────────────────────

    async def create_runner_resources(self) -> tuple[int, str]:
        """Steps 1-3: provider, compaction enable-assert, runner row.

        Returns ``(runner_id, runner_token)``. The token plaintext is
        returned exactly once by ``runner.create`` and never logged.
        """
        await self.ensure_provider()
        await self.ensure_compaction_enabled()
        runner_id, runner_token = await self.create_runner()
        return runner_id, runner_token

    async def ensure_provider(self) -> int:
        if self._provider_id is None:
            self._provider_id = await self._find_or_create_provider()
        # apiProvider.update is the only RPC accepting contextSize, and
        # reused rows created before this adaptor version lack it — assert
        # the compaction gate on every setup. Idempotent: fires once per
        # ensure_provider() call site (twice per trial by design).
        # contextSize stays unconditional; the optional settings are sent
        # only when configured — an absent key lets the backend preserve
        # the stored row value, while an explicit null would overwrite it
        # (and null reasoningEffort breaks openai_compatible providers).
        payload: dict = {
            "id": self._provider_id,
            "contextSize": self._config.llm_context_size,
        }
        if self._config.llm_supports_image is not None:
            payload["supportsImage"] = self._config.llm_supports_image
        if self._config.llm_temperature is not None:
            payload["temperature"] = self._config.llm_temperature
        if self._config.llm_reasoning_effort is not None:
            payload["reasoningEffort"] = self._config.llm_reasoning_effort
        await self._client.call("apiProvider.update", payload)
        return self._provider_id

    async def _find_or_create_provider(self) -> int:
        name = f"{PROVIDER_NAME_PREFIX}{self._config.llm_type}"
        listing = await self._client.call("apiProvider.listAll", {})
        for item in listing.get("items", []):
            if item.get("name") == name:
                self._logger.info("reusing api provider %r (id=%s)", name, item["id"])
                return int(item["id"])
        try:
            created = await self._client.call(
                "apiProvider.create",
                {
                    "name": name,
                    "type": self._config.llm_type,
                    "baseUrl": self._config.llm_base_url,
                    "apiKey": self._config.llm_api_key,
                    "modelId": self._config.llm_model_id,
                },
            )
        except BackendRpcError:
            # Concurrent first-trial create race: another trial created the
            # row between our list and create. Re-list and reuse.
            self._logger.info("provider create raced; re-listing", exc_info=True)
            listing = await self._client.call("apiProvider.listAll", {})
            for item in listing.get("items", []):
                if item.get("name") == name:
                    return int(item["id"])
            raise
        return int(created["id"])

    async def ensure_compaction_enabled(self) -> None:
        # Idempotent. Writes 'true' (never deletes) because an explicit
        # 'false' row — written by adaptor versions before compaction was
        # enabled — overrides the backend seed default forever.
        await self._client.call(
            "config.set",
            {"key": COMPACTION_CONFIG_KEY, "value": "true"},
        )

    async def create_runner(self) -> tuple[int, str]:
        response = await self._client.call(
            "runner.create", {"name": self._runner_name()}
        )
        return int(response["runner"]["id"]), str(response["token"])

    # ── step 4: wait for the in-container runner ──────────────────────────

    async def wait_runner_online(
        self,
        runner_id: int,
        *,
        timeout_sec: float | None = None,
        fetch_log_tail: Callable[[], Awaitable[str]] | None = None,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + (
            timeout_sec if timeout_sec is not None
            else self._config.runner_online_timeout_sec
        )
        while True:
            listing = await self._client.call("runner.list", {})
            for item in listing.get("items", []):
                if item.get("id") == runner_id:
                    if item.get("status") == "online":
                        return
                    break
            if asyncio.get_running_loop().time() >= deadline:
                limit = (
                    timeout_sec
                    if timeout_sec is not None
                    else self._config.runner_online_timeout_sec
                )
                raise ProvisioningError(
                    f"runner {runner_id} ({self._session_id}) did not come "
                    f"online within {limit}s"
                    f"{await self._log_tail_snippet(fetch_log_tail)}"
                )
            await asyncio.sleep(RUNNER_POLL_INTERVAL_SEC)

    async def _log_tail_snippet(
        self, fetch_log_tail: Callable[[], Awaitable[str]] | None
    ) -> str:
        if fetch_log_tail is None:
            return ""
        try:
            tail = await fetch_log_tail()
        except Exception:  # noqa: BLE001 - diagnostics must never mask the error
            self._logger.warning("failed to fetch runner.log tail", exc_info=True)
            return ""
        if not tail.strip():
            return ""
        lines = tail.strip().splitlines()[-RUNNER_LOG_TAIL_LINES:]
        return "\n--- runner.log tail ---\n" + "\n".join(lines)

    # ── steps 5-8: project resources once the runner is online ───────────

    async def create_project_resources(
        self, runner_id: int, runner_token: str
    ) -> TrialResources:
        tool_defs = await self.fetch_tool_definitions()
        provider_id = await self.ensure_provider()
        project_id = await self.create_project()
        await self.assign_runner(project_id, runner_id)
        source_folder_id = await self.create_source_folder(project_id)
        bot_id = await self.create_bot(project_id, provider_id)
        await self.allow_headless_execution(project_id, tool_defs)

        self._resources = TrialResources(
            provider_id=provider_id,
            runner_id=runner_id,
            runner_token=runner_token,
            project_id=project_id,
            source_folder_id=source_folder_id,
            bot_id=bot_id,
        )
        return self._resources

    async def create_project(self) -> int:
        response = await self._client.call(
            "project.create", {"name": self._project_name()}
        )
        return int(response["id"])

    async def assign_runner(self, project_id: int, runner_id: int) -> None:
        # runnerId must never be null — NULL resolves to the backend's own
        # host-local runner, whose filesystem is not the task container's.
        await self._client.call(
            "project.assignRunner",
            {"projectId": project_id, "runnerId": runner_id},
        )

    async def create_source_folder(self, project_id: int) -> int:
        response = await self._client.call(
            "project.sourceFolder.create",
            {"projectId": project_id, "sourcePath": self._config.container_workdir},
        )
        return int(response["id"])

    async def create_bot(self, project_id: int, provider_id: int) -> int:
        response = await self._client.call(
            "bot.create",
            {
                "name": BOT_NAME,
                "projectId": project_id,
                "apiProviderId": provider_id,
                "tools": INCLUDED_TOOLS,
            },
        )
        return int(response["id"])

    async def fetch_tool_definitions(self) -> list[dict]:
        response = await self._client.call("tool.list", {})
        return list(response.get("items", []))

    async def allow_headless_execution(
        self, project_id: int, tool_definitions: list[dict]
    ) -> None:
        """Pin INCLUDED_TOOLS and disable approval defaults for included tools.

        An unanswered approval request hangs an unattended stream until
        harbor's timeout, so every included tool whose
        ``defaultNeedsApproval`` is true gets a per-project override.
        Deriving the set from ``tool.list`` (fetched once) keeps this
        correct if a future backend release adds approval-defaulting
        tools.
        """
        ids = [str(item["id"]) for item in tool_definitions]
        prefix = derive_tool_namespace_prefix(ids)
        base_ids = {strip_tool_namespace(i, prefix) for i in ids}

        missing = [t for t in INCLUDED_TOOLS if t not in base_ids]
        if missing:
            raise ProvisioningError(
                f"backend tool.list is missing expected tools {missing} "
                f"(prefix {prefix!r} derived from response) — the bot's "
                "toolset would silently shrink; refusing to continue"
            )

        by_base_id = {
            strip_tool_namespace(str(item["id"]), prefix): item
            for item in tool_definitions
        }
        for tool_id in INCLUDED_TOOLS:
            if by_base_id[tool_id].get("defaultNeedsApproval"):
                await self._client.call(
                    "tool.permission.set",
                    {"projectId": project_id, "toolId": tool_id, "needsApproval": False},
                )

    # ── naming ────────────────────────────────────────────────────

    def _runner_name(self) -> str:
        return f"harbor-{self._session_id}-{self._suffix}"

    def _project_name(self) -> str:
        return f"harbor-{self._session_id}-{self._suffix}"
