"""Provisioning-flow tests against a scripted stateful fake client."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import pytest

from slopon_harbor.backend_client import BackendRpcError
from slopon_harbor.config import AdaptorConfig
from slopon_harbor.provisioning import (
    INCLUDED_TOOLS,
    TrialProvisioner,
    derive_tool_namespace_prefix,
    strip_tool_namespace,
)

NAMESPACE = "noslop_v1_"


def fake_tool_list(approval_tools: set[str]) -> dict:
    return {
        "items": [
            {
                "id": f"{NAMESPACE}{tid}",
                "name": tid,
                "description": f"tool {tid}",
                "configurable": False,
                "defaultNeedsApproval": tid in approval_tools,
                "isClientForwarded": False,
            }
            for tid in INCLUDED_TOOLS + ["ask_question", "delegate_task"]
        ]
    }


@dataclass
class FakeClient:
    calls: list[tuple[str, dict]] = field(default_factory=list)
    providers: list[dict] = field(default_factory=list)
    runners: list[dict] = field(default_factory=list)
    projects: list[dict] = field(default_factory=list)
    source_folders: list[dict] = field(default_factory=list)
    bots: list[dict] = field(default_factory=list)
    permissions: list[dict] = field(default_factory=list)
    config_rows: dict[str, str] = field(default_factory=dict)
    tool_list_response: dict = field(
        default_factory=lambda: fake_tool_list({"execute_command", "delete_file"})
    )
    runner_status_override: dict[int, str] = field(default_factory=dict)
    runner_ids_that_go_online: set[int] = field(default_factory=set)
    fail_provider_create_once: bool = False

    def _next_id(self, rows: list) -> int:
        return len(rows) + 1

    async def call(self, method: str, params: dict) -> dict:
        self.calls.append((method, dict(params)))
        if method == "apiProvider.listAll":
            return {"items": [dict(p) for p in self.providers]}
        if method == "apiProvider.create":
            if self.fail_provider_create_once:
                self.fail_provider_create_once = False
                # Simulate the race: the competing trial's row lands
                # between our list and our create.
                self.providers.append(
                    {"id": 77, "name": params["name"]}
                )
                raise BackendRpcError(
                    "apiProvider.create", "DUPLICATE", "already exists"
                )
            row = {"id": self._next_id(self.providers), **params}
            self.providers.append(row)
            return dict(row)
        if method == "apiProvider.update":
            for row in self.providers:
                if row["id"] == params["id"]:
                    row.update(params)
                    return dict(row)
            raise AssertionError(
                f"apiProvider.update for unknown id {params.get('id')!r}"
            )
        if method == "config.set":
            self.config_rows[params["key"]] = params["value"]
            return {"key": params["key"], "value": params["value"]}
        if method == "runner.create":
            runner_id = self._next_id(self.runners)
            self.runners.append({"id": runner_id, "name": params["name"]})
            return {"runner": {"id": runner_id, "name": params["name"]}, "token": "tok"}
        if method == "runner.list":
            items = [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "status": self.runner_status_override.get(r["id"], "offline"),
                }
                for r in self.runners
            ]
            return {"items": items}
        if method == "project.create":
            project_id = self._next_id(self.projects)
            self.projects.append({"id": project_id, "name": params["name"]})
            return {"id": project_id, "name": params["name"]}
        if method == "project.assignRunner":
            for project in self.projects:
                if project["id"] == params["projectId"]:
                    project["runnerId"] = params["runnerId"]
            return {"id": params["projectId"]}
        if method == "project.sourceFolder.create":
            folder_id = self._next_id(self.source_folders)
            self.source_folders.append({"id": folder_id, **params})
            return {"id": folder_id, **params}
        if method == "bot.create":
            bot_id = self._next_id(self.bots)
            self.bots.append({"id": bot_id, **params})
            return {"id": bot_id, **params}
        if method == "tool.list":
            return self.tool_list_response
        if method == "tool.permission.set":
            self.permissions.append(dict(params))
            return dict(params)
        raise AssertionError(f"unexpected method {method}")

    def methods(self, name: str) -> list[dict]:
        return [params for method, params in self.calls if method == name]


@pytest.fixture
def config() -> AdaptorConfig:
    return AdaptorConfig(
        backend_url="ws://10.0.0.1:4200",
        backend_api_key="k" * 40,
        backend_port=4200,
        backend_public_url=None,
        runner_runtime=__import__("pathlib").Path("/runtime"),
        container_workdir="/app",
        llm_type="openai_compatible",
        llm_base_url="https://llm.example.com/v1",
        llm_api_key="llm-key",
        llm_model_id="model-x",
        llm_context_size=128000,
        runner_online_timeout_sec=0.3,
        node_version="v22.17.1",
    )


def make_provisioner(client, config, session_id="hello-world__abc123__agent"):
    return TrialProvisioner(
        client, config, session_id, logger=logging.getLogger("test")
    )


async def full_flow(client, config, *, online=True):
    provisioner = make_provisioner(client, config)
    runner_id, token = await provisioner.create_runner_resources()
    client.runner_status_override[runner_id] = "online" if online else "offline"
    await provisioner.wait_runner_online(runner_id, timeout_sec=0.5)
    resources = await provisioner.create_project_resources(runner_id, token)
    return resources


class TestProvider:
    async def test_provider_created_once_and_reused(self, config):
        client = FakeClient()
        await full_flow(client, config)
        await full_flow(client, config)
        # First flow creates; second flow finds by name and reuses.
        assert len(client.methods("apiProvider.create")) == 1
        created = client.methods("apiProvider.create")[0]
        assert created["name"] == "benchmark-openai_compatible"
        assert created["baseUrl"] == "https://llm.example.com/v1"
        assert created["modelId"] == "model-x"
        # reasoningEffort must never be sent (backend defaults it).
        assert "reasoningEffort" not in created

    async def test_provider_create_race_recovers(self, config):
        client = FakeClient()
        client.fail_provider_create_once = True
        provisioner = make_provisioner(client, config)
        provider_id = await provisioner.ensure_provider()
        assert provider_id == 77
        assert len(client.methods("apiProvider.create")) == 1
        # The contextSize gate is asserted on the raced row too.
        assert client.methods("apiProvider.update") == [
            {"id": 77, "contextSize": 128000}
        ]

    async def test_compaction_enabled_config_written(self, config):
        client = FakeClient()
        await full_flow(client, config)
        assert client.config_rows.get("global.context-compaction.enabled") == "true"

    async def test_provider_context_size_asserted(self, config):
        client = FakeClient()
        resources = await full_flow(client, config)
        expected = {
            "id": resources.provider_id,
            "contextSize": config.llm_context_size,
        }
        # Params asserted, not an exact count: the update legitimately
        # fires on both ensure_provider call sites per trial.
        assert expected in client.methods("apiProvider.update")

    async def test_provider_context_size_reasserted_on_reuse(self, config):
        client = FakeClient()
        await full_flow(client, config)
        updates_after_first = len(client.methods("apiProvider.update"))
        assert updates_after_first >= 1
        await full_flow(client, config)
        updates = client.methods("apiProvider.update")
        assert len(updates) > updates_after_first
        provider_id = client.providers[0]["id"]
        for update in updates:
            assert update == {"id": provider_id, "contextSize": 128000}


class TestRunnerAndProject:
    async def test_names_share_session_and_suffix(self, config):
        client = FakeClient()
        await full_flow(client, config)
        runner_name = client.runners[0]["name"]
        project_name = client.projects[0]["name"]
        assert runner_name.startswith("harbor-hello-world__abc123__agent-")
        assert project_name.startswith("harbor-hello-world__abc123__agent-")
        suffix = runner_name.rsplit("-", 1)[1]
        assert len(suffix) == 4
        assert runner_name == project_name

    async def test_same_session_rerun_gets_distinct_names(self, config):
        client = FakeClient()
        await full_flow(client, config)
        await full_flow(client, config)
        assert len({r["name"] for r in client.runners}) == 2
        assert len({p["name"] for p in client.projects}) == 2

    async def test_runner_online_timeout_includes_log_tail(self, config):
        client = FakeClient()

        async def fetch_tail():
            return "line1\nERR_MODULE_NOT_FOUND: ws\nline3"

        provisioner = make_provisioner(client, config)
        runner_id, _token = await provisioner.create_runner_resources()
        client.runner_status_override[runner_id] = "offline"
        with pytest.raises(Exception, match="did not come online") as excinfo:
            await provisioner.wait_runner_online(
                runner_id, timeout_sec=0.0, fetch_log_tail=fetch_tail
            )
        assert "ERR_MODULE_NOT_FOUND" in str(excinfo.value)

    async def test_runner_online_poll_succeeds(self, config):
        client = FakeClient()
        provisioner = make_provisioner(client, config)
        runner_id, _token = await provisioner.create_runner_resources()

        async def go_online():
            await asyncio.sleep(0.05)
            client.runner_status_override[runner_id] = "online"

        await asyncio.gather(
            provisioner.wait_runner_online(runner_id, timeout_sec=2.0), go_online()
        )

    async def test_assignment_never_null(self, config):
        client = FakeClient()
        await full_flow(client, config)
        assignments = client.methods("project.assignRunner")
        assert len(assignments) == 1
        assert assignments[0]["runnerId"] == client.runners[0]["id"]
        assert assignments[0]["runnerId"] is not None

    async def test_source_folder_uses_container_workdir(self, config):
        client = FakeClient()
        resources = await full_flow(client, config)
        created = client.methods("project.sourceFolder.create")[0]
        assert created == {
            "projectId": resources.project_id,
            "sourcePath": "/app",
        }

    async def test_bot_uses_base_tool_ids(self, config):
        client = FakeClient()
        resources = await full_flow(client, config)
        created = client.methods("bot.create")[0]
        assert created["name"] == "benchmark"
        assert created["projectId"] == resources.project_id
        assert created["apiProviderId"] == resources.provider_id
        assert created["tools"] == INCLUDED_TOOLS
        assert "ask_question" not in created["tools"]
        for tool in created["tools"]:
            assert not tool.startswith(NAMESPACE)


class TestToolPinAndPermissions:
    async def test_tool_list_fetched_exactly_once(self, config):
        client = FakeClient()
        await full_flow(client, config)
        assert len(client.methods("tool.list")) == 1

    async def test_missing_tool_fails_loudly(self, config):
        client = FakeClient()
        response = fake_tool_list({"execute_command", "delete_file"})
        response["items"] = [
            item
            for item in response["items"]
            if item["id"] != f"{NAMESPACE}lsp_find_references"
        ]
        client.tool_list_response = response
        provisioner = make_provisioner(client, config)
        runner_id, token = await provisioner.create_runner_resources()
        with pytest.raises(Exception, match="lsp_find_references"):
            await provisioner.create_project_resources(runner_id, token)

    async def test_approval_overrides_derived(self, config):
        client = FakeClient()
        await full_flow(client, config)
        permissions = client.methods("tool.permission.set")
        assert len(permissions) == 2
        by_tool = {p["toolId"]: p for p in permissions}
        assert set(by_tool) == {"execute_command", "delete_file"}
        for perm in permissions:
            assert perm["needsApproval"] is False
            assert perm["projectId"] == client.projects[0]["id"]

    async def test_no_overrides_when_no_approval_defaults(self, config):
        client = FakeClient()
        client.tool_list_response = fake_tool_list(set())
        await full_flow(client, config)
        assert client.methods("tool.permission.set") == []


class TestNamespaceDerivation:
    def test_prefix_derived_from_response(self):
        ids = [f"{NAMESPACE}{t}" for t in INCLUDED_TOOLS]
        assert derive_tool_namespace_prefix(ids) == NAMESPACE

    def test_prefix_changes_with_backend(self):
        ids = [f"other_v9_{t}" for t in INCLUDED_TOOLS[:3]]
        assert derive_tool_namespace_prefix(ids) == "other_v9_"

    def test_no_common_prefix(self):
        assert derive_tool_namespace_prefix(["alpha_tool", "beta_tool"]) == ""
        assert derive_tool_namespace_prefix([]) == ""

    def test_strip(self):
        assert strip_tool_namespace(f"{NAMESPACE}read_file", NAMESPACE) == "read_file"
        assert strip_tool_namespace("read_file", NAMESPACE) == "read_file"
