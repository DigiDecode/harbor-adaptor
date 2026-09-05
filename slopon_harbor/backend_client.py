"""Async WebSocket JSON-RPC client for the SlopOn backend.

Mirrors the wire behavior of the reference client in
``/backend/.e2e/driver.mjs`` against the message envelope defined in
``/backend/src/rpc/types/messages.ts``:

- client sends ``{type: 'auth', apiKey}`` immediately after connect;
- requests carry string ids; responses/errors correlate by id;
- ``chunk`` messages belong to the request id of a streaming call;
- server-initiated ``request`` messages (pushes) are forwarded to the
  optional ``on_push`` callback — invoked synchronously on the reader
  loop, so it must be fast/non-blocking (record-and-log only) — and
  answered with an empty response ``{}`` (the backend treats that as
  "handled");
- unsolicited ``event`` pushes are logged and dropped.

Auth failure is detected explicitly: the backend replies
``{id: '', code: 'AUTH_INVALID'}`` and closes the socket before any
handshake acknowledgement would arrive, so an error with an empty id or a
close observed before the handshake settles raises
:class:`BackendAuthError` instead of a generic connection loss.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
from collections.abc import Callable
from typing import Any

import websockets

# The reference Node client (`ws`) caps inbound frames at 100 MiB by
# default; mirror that instead of the websockets library's 1 MiB default,
# which would corrupt large `chat.getHistory` responses.
_MAX_FRAME_BYTES = 100 * 1024 * 1024

ChunkCallback = Callable[[Any], None]
PushCallback = Callable[[str, dict[str, Any]], None]


class BackendClientError(Exception):
    """Base class for backend client failures."""


class BackendAuthError(BackendClientError):
    """The backend rejected our API key (or closed before the handshake)."""


class BackendTimeoutError(BackendClientError):
    """A call/stream exceeded its client-side deadline."""

    def __init__(self, method: str, timeout: float):
        super().__init__(f"TIMEOUT {method} after {timeout}s")
        self.method = method
        self.timeout = timeout


class BackendConnectionLostError(BackendClientError):
    """The connection dropped while calls were pending."""

    def __init__(self):
        super().__init__("CONNECTION_LOST")


class BackendRpcError(BackendClientError):
    """The backend answered a request with an ``error`` message."""

    def __init__(self, method: str, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.method = method
        self.code = code
        self.message = message


class _Pending:
    __slots__ = ("rid", "method", "future")

    def __init__(self, rid: str, method: str):
        self.rid = rid
        self.method = method
        self.future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()


class SlopOnBackendClient:
    """One WebSocket connection to a SlopOn backend instance."""

    def __init__(
        self,
        url: str,
        api_key: str,
        *,
        logger: logging.Logger,
        auth_settle_sec: float = 0.5,
        on_push: PushCallback | None = None,
    ):
        self._url = url
        self._api_key = api_key
        self._logger = logger
        self._auth_settle_sec = auth_settle_sec
        # Invoked synchronously on the reader loop for every server-initiated
        # request; must be fast/non-blocking (same contract as on_chunk).
        self._on_push = on_push
        self._ws: websockets.ClientConnection | None = None
        self._next_id = itertools.count(1)
        self._pending: dict[str, _Pending] = {}
        self._stream_on_chunk: dict[str, ChunkCallback] = {}
        self._authenticated = False
        self._reader: asyncio.Task[None] | None = None
        # Set (with an exception) only when auth fails; never set on success.
        self._auth_failed: asyncio.Future[BackendClientError] | None = None

    # ── lifecycle ────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the socket and complete the auth handshake.

        Raises :class:`BackendAuthError` when the backend signals
        ``AUTH_INVALID`` or closes the socket before the handshake settles.
        """
        if self._ws is not None:
            raise BackendClientError("client already connected")
        try:
            self._ws = await websockets.connect(
                self._url, max_size=_MAX_FRAME_BYTES
            )
        except BackendClientError:
            raise
        except Exception as err:
            raise BackendClientError(
                f"failed to connect to backend at {self._url}: {err}"
            ) from err

        loop = asyncio.get_running_loop()
        self._auth_failed = loop.create_future()
        self._reader = loop.create_task(self._read_loop())
        # The backend never acknowledges successful auth; it only reacts to
        # failure. Observe the socket for a short settle window: an
        # AUTH_INVALID error or an early close arrives in that window,
        # anything else means the handshake went through.
        try:
            await self._ws.send(
                json.dumps({"type": "auth", "apiKey": self._api_key})
            )
        except Exception as err:
            raise BackendAuthError(f"failed to send auth message: {err}") from err
        try:
            # A failure resolves the future with an exception, which this
            # await re-raises directly.
            await asyncio.wait_for(
                asyncio.shield(self._auth_failed), timeout=self._auth_settle_sec
            )
        except TimeoutError:
            self._authenticated = True
            return
        # Defensive: _auth_failed never resolves successfully.
        raise BackendAuthError("authentication failed")

    async def close(self) -> None:
        """Close the connection and reject any still-pending calls."""
        if self._reader is not None:
            self._reader.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                self._logger.debug("backend client close raised", exc_info=True)
        if self._reader is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader
        self._reject_pending(BackendConnectionLostError())

    # ── RPC surface ──────────────────────────────────────────────────────

    async def call(
        self, method: str, params: dict[str, Any], *, timeout: float | None = 30.0
    ) -> dict[str, Any]:
        """Send a request and await its terminal response/error."""
        pending = self._register_pending(method)
        await self._send_request(pending, method, params)
        try:
            result = await self._await_pending(pending, timeout)
        except asyncio.CancelledError:
            self._drop_pending(pending)
            raise
        return result

    async def stream(
        self,
        method: str,
        params: dict[str, Any],
        *,
        on_chunk: ChunkCallback | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a streaming request; route ``chunk`` messages to ``on_chunk``.

        ``timeout=None`` (the default) means no client deadline: harbor's
        ``asyncio.wait_for`` owns cancellation of the agent phase.
        """
        pending = self._register_pending(method)
        if on_chunk is not None:
            self._stream_on_chunk[pending.rid] = on_chunk
        await self._send_request(pending, method, params)
        try:
            return await self._await_pending(pending, timeout)
        except asyncio.CancelledError:
            # The caller (harbor) timed out or aborted. Drop the pending
            # registration but keep the connection usable for follow-ups
            # (chat.stopStream / chat.getHistory on cancellation paths).
            self._drop_pending(pending)
            raise

    # ── internals ────────────────────────────────────────────────────────

    def _register_pending(self, method: str) -> _Pending:
        if self._ws is None or not self._authenticated:
            raise BackendClientError("client is not connected")
        rid = str(next(self._next_id))
        pending = _Pending(rid, method)
        self._pending[rid] = pending
        return pending

    @staticmethod
    def _id_of(pending: _Pending) -> str:
        return pending.rid

    async def _send_request(
        self, pending: _Pending, method: str, params: dict[str, Any]
    ) -> None:
        rid = self._id_of(pending)
        try:
            await self._ws.send(
                json.dumps(
                    {"type": "request", "id": rid, "method": method, "params": params}
                )
            )
        except Exception as err:
            self._drop_pending(pending)
            raise BackendConnectionLostError() from err

    async def _await_pending(
        self, pending: _Pending, timeout: float | None
    ) -> Any:
        if timeout is None:
            return await pending.future
        try:
            return await asyncio.wait_for(asyncio.shield(pending.future), timeout)
        except TimeoutError:
            self._drop_pending(pending)
            raise BackendTimeoutError(pending.method, timeout) from None

    def _drop_pending(self, pending: _Pending) -> None:
        self._pending.pop(self._id_of(pending), None)
        self._stream_on_chunk.pop(self._id_of(pending), None)

    def _reject_pending(self, error: Exception) -> None:
        for pending in list(self._pending.values()):
            if not pending.future.done():
                pending.future.set_exception(error)
        self._pending.clear()
        self._stream_on_chunk.clear()

    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    self._logger.warning("backend sent non-JSON message; ignoring")
                    continue
                self._handle_message(msg)
        except websockets.ConnectionClosed:
            pass
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - reader must never crash silently
            self._logger.exception("backend client reader crashed")
        finally:
            self._on_disconnected()

    def _handle_message(self, msg: dict[str, Any]) -> None:
        mtype = msg.get("type")
        if mtype == "chunk":
            on_chunk = self._stream_on_chunk.get(msg.get("id"))
            if on_chunk is not None:
                try:
                    on_chunk(msg.get("data"))
                except Exception:  # noqa: BLE001 - chunk sink must not kill reader
                    self._logger.exception("on_chunk callback raised")
            return
        if mtype in ("response", "error"):
            rid = msg.get("id")
            pending = self._pending.get(rid)
            if pending is None:
                # An error with an empty id is the backend's auth-rejection
                # signal; it correlates to no request by construction.
                if mtype == "error" and not rid:
                    self._fail_auth(
                        BackendAuthError(
                            f"backend rejected authentication: "
                            f"{msg.get('code')}: {msg.get('message')}"
                        )
                    )
                else:
                    self._logger.debug(
                        "response/error for unknown id %r ignored", rid
                    )
                return
            self._pending.pop(rid, None)
            self._stream_on_chunk.pop(rid, None)
            if mtype == "response":
                if not pending.future.done():
                    pending.future.set_result(msg.get("data") or {})
            elif not pending.future.done():
                pending.future.set_exception(
                    BackendRpcError(
                        pending.method,
                        str(msg.get("code") or "UNKNOWN"),
                        str(msg.get("message") or ""),
                    )
                )
            return
        if mtype == "request":
            # Server-initiated push (e.g. tool approval requests to the
            # desktop client, compaction lifecycle notifications to this
            # adaptor). Forward to the registered handler, then answer with
            # an empty response exactly as the reference client does for
            # unknown methods (the backend treats that as "handled").
            method = msg.get("method")
            if self._on_push is not None and method is not None:
                try:
                    self._on_push(method, msg.get("params") or {})
                except Exception:  # noqa: BLE001 - push sink must not kill reader
                    self._logger.exception("on_push callback raised")
            self._logger.debug(
                "backend pushed request %s; replying with empty response",
                method,
            )
            reply = json.dumps({"type": "response", "id": msg.get("id"), "data": {}})
            task = asyncio.get_running_loop().create_task(self._ws.send(reply))
            task.add_done_callback(self._log_send_failure)
            return
        if mtype == "event":
            self._logger.debug("backend event: %s", msg.get("event"))
            return
        self._logger.debug("ignoring backend message of type %r", mtype)

    @staticmethod
    def _log_send_failure(task: asyncio.Task[None]) -> None:
        if not task.cancelled() and task.exception() is not None:
            logging.getLogger(__name__).debug(
                "failed to reply to server push", exc_info=task.exception()
            )

    def _fail_auth(self, error: BackendClientError) -> None:
        if self._auth_failed is not None and not self._auth_failed.done():
            self._auth_failed.set_exception(error)
            # Keep the exception from being reported as "never retrieved".
            self._auth_failed.exception()

    def _on_disconnected(self) -> None:
        if self._authenticated:
            self._reject_pending(BackendConnectionLostError())
            return
        self._fail_auth(
            BackendAuthError("backend closed the connection during auth handshake")
        )
