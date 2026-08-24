"""Agent lifecycle tests with injected client + environment doubles."""

from __future__ import annotations

import asyncio
import json
import logging
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from slopon_harbor.agent import SlopOnAgent
from slopon_harbor.backend_client import BackendRpcError

NAMESPACE = "noslop_v1_"
INCLUDED_TOOLS = [
    "list_directory", "read_file", "search_files", "search_content",
    "write_file", "edit_file", "move_file", "delete_file",
    "execute_command", "manage_process", "lsp_goto_definition",
    "lsp_find_references",
]


# ── doubles ──────────────────────────────────────────────────────────────


@dataclass
class FakeBackendForAgent:
    """Client double implementing the exact RPC surface the agent uses."""

    calls: list[tuple[str, dict]] = field(default_factory=list)
    closes: int = 0
    connects: int = 0
    stream_gate: asyncio.Event = field(default_factory=asyncio.Event)
    stream_error: Exception | None = None
    chats: list[int] = field(default_factory=list)

    async def connect(self):
        self.connects += 1

    async def close(self):
        self.closes += 1

    async def call(self, method, params, *, timeout=None):
        self.calls.append((method, dict(params)))
        if method == "apiProvider.listAll":
            return {"items": []}
        if method == "apiProvider.create":
            return {"id": 1}
        if method == "config.set":
            return {}
        if method == "runner.create":
            return {"runner": {"id": 5, "name": "harbor-x-a1b2"}, "token": "RUNNER-TOKEN"}
        if method == "runner.list":
            return {"items": [{"id": 5, "name": "harbor-x-a1b2", "status": "online"}]}
        if method == "project.create":
            return {"id": 11, "name": "harbor-x-a1b2"}
        if method == "project.assignRunner":
            return {"id": 11}
        if method == "project.sourceFolder.create":
            return {"id": 21}
        if method == "bot.create":
            return {"id": 31}
        if method == "tool.list":
            return {
                "items": [
                    {
                        "id": f"{NAMESPACE}{tid}",
                        "defaultNeedsApproval": tid
                        in ("execute_command", "delete_file"),
                    }
                    for tid in INCLUDED_TOOLS + ["ask_question"]
                ]
            }
        if method == "tool.permission.set":
            return {}
        if method == "chat.create":
            chat_id = len(self.chats) + 100
            self.chats.append(chat_id)
            return {"id": chat_id}
        if method == "chat.stopStream":
            return {"stopped": True}
        if method == "chat.getHistory":
            return {
                "chatId": params["chatId"],
                "messages": [{"role": "user", "content": "hello"}],
            }
        if method == "chat.getTokenUsage":
            return {
                "chatId": params["chatId"],
                "inputTokens": 10,
                "outputTokens": 5,
                "cacheHitTokens": 2,
                "totalTokens": 17,
                "contextSize": None,
            }
        raise AssertionError(f"unexpected method {method}")

    def methods(self, name):
        return [params for method, params in self.calls if method == name]

    async def stream(self, method, params, *, on_chunk=None, timeout=None):
        self.calls.append((method, dict(params)))
        if self.stream_error is not None:
            raise self.stream_error
        # Simulate a chunk then wait for the test to end the stream.
        if on_chunk is not None:
            on_chunk({"text": "delta"})
        await self.stream_gate.wait()
        return {"chatId": params["chatId"], "messageId": "m1"}


@dataclass
class ExecRecord:
    command: str
    env: dict | None


class FakeEnvironment:
    """Duck-typed BaseEnvironment recording execs and uploads."""

    def __init__(self, storage: Path | None = None):
        self.execs: list[ExecRecord] = []
        self.uploads: list[tuple[Path, str]] = []
        self.gateway = "172.17.0.1"
        self.node_bin = "/tmp/slopon-runner/node/bin/node"
        # The agent deletes its temp tarball after upload; copy uploads so
        # tests can inspect the content afterwards.
        self._storage = storage or Path(tempfile.mkdtemp(prefix="fake-env-uploads-"))

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.execs.append(ExecRecord(command=command, env=dict(env) if env else None))
        if "nodejs.org" in command or "node -v" in command:
            return SimpleNamespace(stdout=self.node_bin + "\n", stderr="", return_code=0)
        if "/proc/net/route" in command:
            return SimpleNamespace(stdout=self.gateway + "\n", stderr="", return_code=0)
        if command.startswith("tail "):
            return SimpleNamespace(stdout="log-tail\n", stderr="", return_code=0)
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    async def upload_file(self, source_path, target_path):
        kept = self._storage / Path(source_path).name
        import shutil

        shutil.copy2(source_path, kept)
        self.uploads.append((kept, target_path))


# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def env_bundle(tmp_path, monkeypatch):
    """Valid HOME config + runtime dir + process LLM key."""
    slopon = tmp_path / ".slopon"
    slopon.mkdir()
    (slopon / "config.json").write_text(
        json.dumps(
            {"server": {"port": 4200, "listenIp": "0.0.0.0", "apiKey": "k" * 40}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SLOPON_LLM_API_KEY", "llm-key")
    for var in (
        "SLOPON_BACKEND_URL",
        "SLOPON_BACKEND_API_KEY",
        "SLOPON_BACKEND_PUBLIC_URL",
    ):
        monkeypatch.delenv(var, raising=False)

    runtime = tmp_path / "slopon-runtime"
    runtime.mkdir()
    (runtime / "runner.js").write_text("runner", encoding="utf-8")
    (runtime / "package.json").write_text("{}", encoding="utf-8")
    (runtime / "prompts").mkdir()
    (runtime / "node_modules").mkdir()
    (runtime / "index.js").write_text("server", encoding="utf-8")
    return SimpleNamespace(runtime=runtime, logs=tmp_path / "agent-logs")


def make_agent(env_bundle) -> SlopOnAgent:
    env_bundle.logs.mkdir(exist_ok=True)
    return SlopOnAgent(
        logs_dir=env_bundle.logs,
        model_name="openai/gpt-test",
        extra_env={
            "SLOPON_RUNNER_RUNTIME": str(env_bundle.runtime),
            "SLOPON_LLM_BASE_URL": "https://llm.example.com/v1",
        },
        logger=logging.getLogger("agent-test"),
    )


@pytest.fixture
def wired_agent(env_bundle, monkeypatch):
    agent = make_agent(env_bundle)
    clients: list[FakeBackendForAgent] = []
    created: list[FakeBackendForAgent] = []

    def factory():
        client = FakeBackendForAgent()
        clients.append(client)
        created.append(client)
        return client

    monkeypatch.setattr(agent, "_connect", factory)
    return SimpleNamespace(agent=agent, clients=clients, env_bundle=env_bundle)


async def complete_setup(agent: SlopOnAgent, environment: FakeEnvironment):
    await agent.setup(environment)


# ── tests ────────────────────────────────────────────────────────────────


class TestSetup:
    async def test_setup_exec_and_upload_sequence(self, wired_agent):
        environment = FakeEnvironment()
        await complete_setup(wired_agent.agent, environment)

        commands = [rec.command for rec in environment.execs]
        node_idx = next(i for i, c in enumerate(commands) if "node -v" in c)
        unpack_idx = next(
            i for i, c in enumerate(commands) if "runtime.tar.gz" in c
        )
        gateway_idx = next(
            i for i, c in enumerate(commands) if "/proc/net/route" in c
        )
        start_idx = next(i for i, c in enumerate(commands) if "nohup" in c)
        assert node_idx < unpack_idx < gateway_idx < start_idx
        # Exactly one upload (the runtime tarball) lands before unpacking.
        assert len(environment.uploads) == 1
        uploaded_path, uploaded_target = environment.uploads[0]
        assert uploaded_target == "/tmp/slopon-runner/runtime.tar.gz"
        # Runner start carries credentials ONLY in the env dict.
        start_rec = environment.execs[start_idx]
        assert start_rec.env == {
            "RUNNER_URL": "ws://172.17.0.1:4200",
            "RUNNER_TOKEN": "RUNNER-TOKEN",
        }
        assert "RUNNER-TOKEN" not in start_rec.command
        assert "ws://" not in start_rec.command
        # Node provisioning and gateway discovery carry no env dict.
        assert environment.execs[node_idx].env is None
        assert environment.execs[gateway_idx].env is None

    async def test_setup_tar_members_exact(self, wired_agent):
        environment = FakeEnvironment()
        await complete_setup(wired_agent.agent, environment)
        uploaded_path, _ = environment.uploads[0]
        with tarfile.open(uploaded_path) as tar:
            names = tar.getnames()
        assert "runner.js" in names
        assert "package.json" in names
        assert "prompts" in names
        assert "node_modules" in names
        assert "index.js" not in names

    async def test_setup_public_url_override(self, wired_agent):
        environment = FakeEnvironment()
        wired_agent.agent._config = wired_agent.agent._config.__class__(
            **{
                **wired_agent.agent._config.__dict__,
                "backend_public_url": "ws://host.docker.internal:4200",
            }
        )
        await complete_setup(wired_agent.agent, environment)
        start_rec = next(r for r in environment.execs if "nohup" in r.command)
        assert start_rec.env["RUNNER_URL"] == "ws://host.docker.internal:4200"
        # Gateway discovery never ran: the URL was configured.
        assert not any("/proc/net/route" in r.command for r in environment.execs)

    async def test_setup_rpcs_and_close(self, wired_agent):
        environment = FakeEnvironment()
        await complete_setup(wired_agent.agent, environment)
        client = wired_agent.clients[0]
        methods = [method for method, _ in client.calls]
        assert "apiProvider.listAll" in methods
        assert "apiProvider.create" in methods
        assert "config.set" in methods
        assert "runner.create" in methods
        assert "project.assignRunner" in methods
        assert "bot.create" in methods
        assert "tool.permission.set" in methods
        assert client.closes == 1
        # Permissions: exactly the two approval-defaulting tools.
        perms = client.methods("tool.permission.set")
        assert {p["toolId"] for p in perms} == {
            "execute_command",
            "delete_file",
        }

    async def test_run_before_setup_raises(self, wired_agent):
        from harbor.models.agent.context import AgentContext

        with pytest.raises(RuntimeError, match="setup"):
            await wired_agent.agent.run(
                "hi", FakeEnvironment(), AgentContext()
            )


class TestRun:
    async def test_run_success_artifacts_and_tokens(self, wired_agent):
        environment = FakeEnvironment()
        await complete_setup(wired_agent.agent, environment)

        client = FakeBackendForAgent()
        wired_agent.clients.append(client)
        wired_agent.agent._connect = lambda: client

        from harbor.models.agent.context import AgentContext

        context = AgentContext()
        client.stream_gate.set()
        await wired_agent.agent.run("do the task", environment, context)

        # Pre-created chat, streamed with the explicit chatId.
        chat_create = client.methods("chat.create")
        assert chat_create == [{"botId": 31}]
        stream_params = client.methods("chat.stream")[0]
        assert stream_params == {
            "chatId": 100,
            "botId": 31,
            "message": "do the task",
        }
        # Token usage lands in the context.
        assert context.n_input_tokens == 10
        assert context.n_cache_tokens == 2
        assert context.n_output_tokens == 5
        # history.json written into logs_dir.
        history = json.loads(
            (wired_agent.env_bundle.logs / "history.json").read_text(encoding="utf-8")
        )
        assert history["chatId"] == 100
        assert history["messages"][0]["role"] == "user"
        assert client.closes == 1

    async def test_two_consecutive_runs_both_succeed(self, wired_agent):
        environment = FakeEnvironment()
        await complete_setup(wired_agent.agent, environment)

        from harbor.models.agent.context import AgentContext

        made = []

        def factory():
            client = FakeBackendForAgent()
            client.stream_gate.set()
            made.append(client)
            return client

        wired_agent.agent._connect = factory
        await wired_agent.agent.run("step one", environment, AgentContext())
        await wired_agent.agent.run("step two", environment, AgentContext())
        # Fresh connection per call (multi-step semantics).
        assert len(made) == 2
        assert all(c.closes == 1 for c in made)
        assert made[0].methods("chat.stream")[0]["message"] == "step one"
        assert made[1].methods("chat.stream")[0]["message"] == "step two"

    async def test_run_rpc_error_writes_error_json(self, wired_agent):
        environment = FakeEnvironment()
        await complete_setup(wired_agent.agent, environment)

        client = FakeBackendForAgent()
        client.stream_error = BackendRpcError(
            "chat.stream", "RUNNER_OFFLINE", "runner is offline"
        )
        wired_agent.agent._connect = lambda: client

        from harbor.models.agent.context import AgentContext

        with pytest.raises(BackendRpcError):
            await wired_agent.agent.run("task", environment, AgentContext())
        payload = json.loads(
            (wired_agent.env_bundle.logs / "error.json").read_text(encoding="utf-8")
        )
        assert payload == {"code": "RUNNER_OFFLINE", "message": "runner is offline"}
        assert client.closes == 1

    async def test_run_cancel_stops_stream_writes_partial_history(self, wired_agent):
        environment = FakeEnvironment()
        await complete_setup(wired_agent.agent, environment)

        client = FakeBackendForAgent()  # stream_gate never set: blocks
        wired_agent.agent._connect = lambda: client

        from harbor.models.agent.context import AgentContext

        task = asyncio.create_task(
            wired_agent.agent.run("long task", environment, AgentContext())
        )
        # Wait until the stream request has actually started.
        while not client.methods("chat.stream"):
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # stopStream was called with the pre-created chatId...
        assert client.methods("chat.stopStream") == [{"chatId": 100}]
        # ...and the partial history was still written.
        history = json.loads(
            (wired_agent.env_bundle.logs / "history.json").read_text(encoding="utf-8")
        )
        assert history["chatId"] == 100
        assert client.closes == 1


class TestAgentInfo:
    def test_to_agent_info_reports_model(self, wired_agent):
        info = wired_agent.agent.to_agent_info()
        assert info.name == "slopon"
        assert info.version
        assert info.model_info is not None
        assert info.model_info.name == "gpt-test"
        assert info.model_info.provider == "openai"

    def test_supports_flags(self, wired_agent):
        agent = wired_agent.agent
        assert agent.SUPPORTS_WINDOWS is False
        assert agent.SUPPORTS_RESUME is False
        assert agent.SUPPORTS_ATIF is False
