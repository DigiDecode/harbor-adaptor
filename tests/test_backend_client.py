"""Wire-protocol tests for SlopOnBackendClient against a fake WS backend."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
import websockets

from slopon_harbor.backend_client import (
    BackendAuthError,
    BackendConnectionLostError,
    BackendRpcError,
    BackendTimeoutError,
    SlopOnBackendClient,
)

API_KEY = "t" * 40
Handler = Callable[[websockets.ServerConnection, dict[str, Any]], Awaitable[None]]


class FakeBackend:
    """In-process backend speaking the SlopOn WS JSON envelope."""

    def __init__(self, handler: Handler, *, expected_key: str = API_KEY):
        self.handler = handler
        self.expected_key = expected_key
        self.received: list[dict[str, Any]] = []
        self.server: websockets.Server | None = None
        self.url = ""

    async def start(self) -> None:
        async def ws_handler(ws: websockets.ServerConnection) -> None:
            async for raw in ws:
                msg = json.loads(raw)
                self.received.append(msg)
                if msg["type"] == "auth":
                    if msg["apiKey"] != self.expected_key:
                        await ws.send(
                            json.dumps(
                                {
                                    "id": "",
                                    "type": "error",
                                    "code": "AUTH_INVALID",
                                    "message": "Invalid API key",
                                }
                            )
                        )
                        await ws.close()
                    continue
                if msg["type"] == "response":
                    continue
                await self.handler(ws, msg)

        self.server = await websockets.serve(ws_handler, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}"

    async def stop(self) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()

    def request_messages(self, method: str) -> list[dict[str, Any]]:
        return [m for m in self.received if m.get("method") == method]


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("slopon-harbor-test")


def make_client(
    url: str, *, auth_settle_sec: float = 0.15, logger: logging.Logger | None = None
) -> SlopOnBackendClient:
    return SlopOnBackendClient(
        url,
        API_KEY,
        logger=logger or logging.getLogger("test"),
        auth_settle_sec=auth_settle_sec,
    )


async def test_auth_handshake_then_call_roundtrip(logger):
    async def handler(ws, msg):
        await ws.send(
            json.dumps(
                {"type": "response", "id": msg["id"], "data": {"items": [1, 2]}}
            )
        )

    fake = FakeBackend(handler)
    await fake.start()
    try:
        client = make_client(fake.url, logger=logger)
        await client.connect()
        result = await client.call("runner.list", {})
        assert result == {"items": [1, 2]}
        assert fake.request_messages("runner.list"), "request must reach backend"
        await client.close()
    finally:
        await fake.stop()


async def test_auth_rejection_raises_backend_auth_error(logger):
    async def handler(ws, msg):  # pragma: no cover - never reached
        await ws.send(json.dumps({"type": "response", "id": msg["id"], "data": {}}))

    fake = FakeBackend(handler, expected_key="wrong-key")
    await fake.start()
    try:
        client = make_client(fake.url, logger=logger)
        with pytest.raises(BackendAuthError, match="AUTH_INVALID"):
            await client.connect()
        await client.close()
    finally:
        await fake.stop()


async def test_close_before_auth_raises_backend_auth_error(logger):
    async def handler(ws, msg):  # pragma: no cover - never reached
        pass

    async def close_immediately(ws):
        await ws.close()

    server = await websockets.serve(close_immediately, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        client = make_client(f"ws://127.0.0.1:{port}", logger=logger)
        with pytest.raises(BackendAuthError, match="auth"):
            await client.connect()
    finally:
        server.close()
        await server.wait_closed()


async def test_call_error_propagates_as_rpc_error(logger):
    async def handler(ws, msg):
        await ws.send(
            json.dumps(
                {
                    "type": "error",
                    "id": msg["id"],
                    "code": "DUPLICATE",
                    "message": "already exists",
                }
            )
        )

    fake = FakeBackend(handler)
    await fake.start()
    try:
        client = make_client(fake.url, logger=logger)
        await client.connect()
        with pytest.raises(BackendRpcError) as excinfo:
            await client.call("runner.create", {"name": "x"})
        assert excinfo.value.code == "DUPLICATE"
        assert excinfo.value.message == "already exists"
        assert excinfo.value.method == "runner.create"
        await client.close()
    finally:
        await fake.stop()


async def test_stream_routes_chunks_and_resolves_on_terminal_response(logger):
    async def handler(ws, msg):
        for i in range(3):
            await ws.send(
                json.dumps({"type": "chunk", "id": msg["id"], "data": {"seq": i}})
            )
        await ws.send(
            json.dumps(
                {"type": "response", "id": msg["id"], "data": {"chatId": 7}}
            )
        )

    fake = FakeBackend(handler)
    await fake.start()
    try:
        client = make_client(fake.url, logger=logger)
        await client.connect()
        chunks: list[Any] = []
        result = await client.stream(
            "chat.stream", {"message": "hi"}, on_chunk=chunks.append, timeout=None
        )
        assert result == {"chatId": 7}
        assert [c["seq"] for c in chunks] == [0, 1, 2]
        await client.close()
    finally:
        await fake.stop()


async def test_server_push_request_gets_empty_response(logger):
    async def handler(ws, msg):
        if msg["method"] == "chat.stream":
            # Push a server-initiated request mid-stream, then terminate.
            await ws.send(
                json.dumps(
                    {
                        "type": "request",
                        "id": "srv-1",
                        "method": "chat.toolApprovalRequest",
                        "params": {"approvalId": "a1"},
                    }
                )
            )
            await ws.send(
                json.dumps({"type": "response", "id": msg["id"], "data": {"ok": True}})
            )

    fake = FakeBackend(handler)
    await fake.start()
    try:
        client = make_client(fake.url, logger=logger)
        await client.connect()
        await client.stream("chat.stream", {}, on_chunk=None, timeout=None)
        # Give the fire-and-forget reply to the server push time to flush
        # before the test tears the connection down.
        await asyncio.sleep(0.2)
        # The fake records our reply to its push in `received`.
        replies = [m for m in fake.received if m.get("type") == "response"]
        assert replies == [{"type": "response", "id": "srv-1", "data": {}}]
        await client.close()
    finally:
        await fake.stop()


async def test_disconnect_after_auth_rejects_pending(logger):
    server_holder: dict[str, websockets.ServerConnection] = {}

    async def handler(ws, msg):
        server_holder["ws"] = ws
        # Never respond; the test closes the socket from the server side.

    fake = FakeBackend(handler)
    await fake.start()
    try:
        client = make_client(fake.url, logger=logger)
        await client.connect()
        with pytest.raises(BackendConnectionLostError, match="CONNECTION_LOST"):
            call_task = asyncio.create_task(client.call("runner.list", {}))
            await asyncio.sleep(0.1)
            await server_holder["ws"].close()
            await call_task
        await client.close()
    finally:
        await fake.stop()


async def test_stream_without_client_deadline_waits_for_slow_terminal(logger):
    async def handler(ws, msg):
        await asyncio.sleep(0.3)
        await ws.send(
            json.dumps({"type": "response", "id": msg["id"], "data": {"done": True}})
        )

    fake = FakeBackend(handler)
    await fake.start()
    try:
        client = make_client(fake.url, logger=logger)
        await client.connect()
        result = await client.stream("chat.stream", {}, timeout=None)
        assert result == {"done": True}
        await client.close()
    finally:
        await fake.stop()


async def test_call_timeout_raises_backend_timeout_error(logger):
    async def handler(ws, msg):  # pragma: no cover - never responds
        await asyncio.sleep(5)

    fake = FakeBackend(handler)
    await fake.start()
    try:
        client = make_client(fake.url, logger=logger)
        await client.connect()
        with pytest.raises(BackendTimeoutError, match="runner.list"):
            await client.call("runner.list", {}, timeout=0.2)
        await client.close()
    finally:
        await fake.stop()


async def test_call_after_close_is_rejected(logger):
    async def handler(ws, msg):  # pragma: no cover
        await ws.send(json.dumps({"type": "response", "id": msg["id"], "data": {}}))

    fake = FakeBackend(handler)
    await fake.start()
    try:
        client = make_client(fake.url, logger=logger)
        await client.connect()
        await client.close()
        from slopon_harbor.backend_client import BackendConnectionLostError

        # Sending on the closed socket surfaces as CONNECTION_LOST (the
        # "not connected" guard only fires before connect()).
        with pytest.raises(BackendConnectionLostError):
            await client.call("runner.list", {})
    finally:
        await fake.stop()
