"""Push-callback registration and response routing of CompanionBase."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any, Callable, Optional

from .base_support import _fmt_path

logger = logging.getLogger("CompanionBase")


class _CallbackMixin:
    """Part of :class:`CompanionBase` (see companion_base.py)."""

    # -------------------------------------------------------------------------
    # Push Callbacks
    # -------------------------------------------------------------------------

    def clear_push_callbacks(self) -> None:
        """Remove all registered push callbacks.

        Called by FrameServer between client connections so that stale
        closures from a previous connection are not invoked on the next one.
        """
        for key in self._push_callbacks:
            self._push_callbacks[key].clear()

    def on_message_event(self, callback: Callable) -> None:
        """Register a direct-message callback receiving one ``MessageEvent``."""
        self._push_callbacks["message_event"].append(callback)

    def on_channel_message_event(self, callback: Callable) -> None:
        """Register a channel-text callback receiving one ``ChannelMessageEvent``."""
        self._push_callbacks["channel_message_event"].append(callback)

    def on_channel_data_event(self, callback: Callable) -> None:
        """Register a channel-data callback receiving one ``ChannelDataEvent``."""
        self._push_callbacks["channel_data_event"].append(callback)

    @staticmethod
    async def _call_legacy(callback: Callable, *args: Any) -> None:
        """Invoke a legacy positional callback, awaiting it when async."""
        result = callback(*args)
        if inspect.isawaitable(result):
            await result

    def on_message_received(self, callback: Callable) -> None:
        """Deprecated: prefer :meth:`on_message_event`.

        The legacy callback receives the ``MessageEvent`` fields exploded
        positionally: ``(sender_key, text, timestamp, txt_type, packet_hash,
        snr, rssi, sender_prefix, path_len, queued)``. The final ``queued``
        flag is false when the protected offline queue could not retain the
        message.
        """

        async def _legacy_adapter(event: Any) -> None:
            await self._call_legacy(
                callback,
                event.sender_key,
                event.text,
                event.timestamp,
                event.txt_type,
                event.packet_hash,
                event.snr,
                event.rssi,
                event.sender_prefix,
                event.path_len,
                event.queued,
            )

        self._push_callbacks["message_event"].append(_legacy_adapter)

    def on_channel_message_received(self, callback: Callable) -> None:
        """Deprecated: prefer :meth:`on_channel_message_event`.

        The legacy callback receives ``(channel_name, sender_name, text,
        timestamp, path_len, channel_idx, packet_hash, snr, rssi, queued)``.
        """

        async def _legacy_adapter(event: Any) -> None:
            await self._call_legacy(
                callback,
                event.channel_name,
                event.sender_name,
                event.text,
                event.timestamp,
                event.path_len,
                event.channel_idx,
                event.packet_hash,
                event.snr,
                event.rssi,
                event.queued,
            )

        self._push_callbacks["channel_message_event"].append(_legacy_adapter)

    def on_channel_data_received(self, callback: Callable) -> None:
        """Deprecated: prefer :meth:`on_channel_data_event`.

        The legacy callback receives ``(channel_idx, path_len, data_type,
        payload, packet_hash, snr, rssi, queued)``.
        """

        async def _legacy_adapter(event: Any) -> None:
            await self._call_legacy(
                callback,
                event.channel_idx,
                event.path_len,
                event.data_type,
                event.payload,
                event.packet_hash,
                event.snr,
                event.rssi,
                event.queued,
            )

        self._push_callbacks["channel_data_event"].append(_legacy_adapter)

    def on_advert_received(self, callback: Callable) -> None:
        self._push_callbacks["advert_received"].append(callback)

    def on_contact_path_updated(self, callback: Callable) -> None:
        self._push_callbacks["contact_path_updated"].append(callback)

    async def _on_contact_path_updated(self, pub: bytes, path_len: int, path_bytes: bytes) -> None:
        """Called by ProtocolResponseHandler when contact's out_path is updated from a PATH packet.

        Matches companion firmware behaviour: PATH updates are only applied
        (and pushed to the client) for contacts that already exist in the
        store.  Unknown public keys are silently ignored.
        """
        contact = self.get_contact_by_key(pub)
        if contact is None:
            logger.debug(
                "[PATHDIAG] _on_contact_path_updated: no contact for pub=%s (ignored)",
                pub[:4].hex(),
            )
            return  # Firmware does not send PATH for non-contacts
        logger.debug(
            "[PATHDIAG] _on_contact_path_updated pub=%s name=%s %s",
            pub[:4].hex(),
            getattr(contact, "name", "?"),
            _fmt_path(path_len, path_bytes),
        )
        contact.out_path_len = path_len
        contact.out_path = path_bytes
        self.contacts.update(contact)
        await self._fire_callbacks("contact_path_updated", contact)

    def on_send_confirmed(self, callback: Callable) -> None:
        self._push_callbacks["send_confirmed"].append(callback)

    def on_trace_received(self, callback: Callable) -> None:
        self._push_callbacks["trace_received"].append(callback)

    def on_node_discovered(self, callback: Callable) -> None:
        self._push_callbacks["node_discovered"].append(callback)

    def on_login_result(self, callback: Callable) -> None:
        self._push_callbacks["login_result"].append(callback)

    def on_telemetry_response(self, callback: Callable) -> None:
        self._push_callbacks["telemetry_response"].append(callback)

    def on_status_response(self, callback: Callable) -> None:
        self._push_callbacks["status_response"].append(callback)

    def on_raw_data_received(self, callback: Callable) -> None:
        self._push_callbacks["raw_data_received"].append(callback)

    def on_rx_log_data(self, callback: Callable) -> None:
        """Register callback for raw RX with SNR/RSSI (CompanionRadio only).

        Callback(snr: float, rssi: int, raw_bytes: bytes). Same data as
        PUSH_CODE_LOG_RX_DATA (0x88). Only fired when using CompanionRadio;
        CompanionBridge does not own the radio.
        """
        self._push_callbacks["rx_log_data"].append(callback)

    def on_binary_response(self, callback: Callable) -> None:
        """Register callback for PUSH 0x8C. Callback(tag_bytes, response_data)."""
        self._push_callbacks["binary_response"].append(callback)

    def on_path_discovery_response(self, callback: Callable) -> None:
        """Register callback for path discovery 0x8D. (tag_bytes, pubkey, out_path, in_path)."""
        self._push_callbacks["path_discovery_response"].append(callback)

    def on_contact_deleted(self, callback: Callable) -> None:
        """Register callback for PUSH 0x8F (contact overwritten). Callback(pub_key_bytes)."""
        self._push_callbacks["contact_deleted"].append(callback)

    def on_contacts_full(self, callback: Callable) -> None:
        """Register callback for PUSH 0x90 (contacts store full). Callback()."""
        self._push_callbacks["contacts_full"].append(callback)

    def on_channel_updated(self, callback: Callable) -> None:
        """Register callback for channel set/remove. Callback(idx: int, channel_or_none)."""
        self._push_callbacks["channel_updated"].append(callback)

    def register_binary_request(
        self,
        tag_hex: str,
        request_type: int,
        timeout_seconds: float,
        pubkey_prefix: str = "",
        context: Optional[dict] = None,
    ) -> None:
        """Register a pending binary request. Call cleanup_expired_requests first."""
        self._pending_binary_requests[tag_hex] = {
            "request_type": request_type,
            "pubkey_prefix": pubkey_prefix,
            "expires_at": time.time() + timeout_seconds,
            "context": context or {},
        }

    def cleanup_expired_binary_requests(self) -> None:
        """Remove expired entries from _pending_binary_requests."""
        now = time.time()
        expired = [
            tag for tag, info in self._pending_binary_requests.items() if now > info["expires_at"]
        ]
        for tag in expired:
            del self._pending_binary_requests[tag]

    async def _on_binary_response(
        self,
        tag_bytes: bytes,
        response_data: bytes,
        path_info: Optional[tuple] = None,
    ) -> None:
        """Called when binary response (tag + data, optional path) received."""
        if path_info is not None:
            if await self._try_handle_path_discovery(tag_bytes, path_info):
                return
        self.cleanup_expired_binary_requests()
        tag_hex = tag_bytes.hex()
        info = self._pending_binary_requests.pop(tag_hex, None)
        if not info:
            # A decryptable response arrived but no request is waiting for this tag.
            # This is the signature of "response arrived but we already timed out"
            # (or a tag mismatch); distinct from "no response arrived at all".
            logger.debug(
                "[PATHDIAG] anon/binary response UNMATCHED tag=%s (%dB) — no pending "
                "request (arrived after timeout, or tag mismatch). pending=%s",
                tag_hex,
                len(response_data),
                list(self._pending_binary_requests.keys()),
            )
            await self._fire_callbacks("binary_response", tag_bytes, response_data)
            return
        request_type = info["request_type"]
        logger.debug(
            "[PATHDIAG] anon/binary response MATCHED tag=%s type=%s (%dB)",
            tag_hex,
            request_type,
            len(response_data),
        )
        pubkey_prefix = info.get("pubkey_prefix", "")
        context = info.get("context", {})
        parsed = None
        try:
            from . import binary_parsing

            parsed = binary_parsing.parse_binary_response(
                request_type,
                response_data,
                pubkey_prefix=pubkey_prefix,
                context=context,
            )
        except Exception as e:
            logger.debug("Binary response parse for type %s: %s", request_type, e)
        await self._fire_callbacks(
            "binary_response", tag_bytes, response_data, parsed, request_type
        )

    async def _try_handle_path_discovery(self, tag_bytes: bytes, path_info: tuple) -> bool:
        """If tag is pending path discovery, fire path_discovery_response and return True."""
        out_len_byte, out_path, in_len_byte, in_path, contact_pubkey = path_info
        tag_int = int.from_bytes(tag_bytes, "little")
        if tag_int not in self._pending_discovery_tags:
            return False
        self._pending_discovery_tags.discard(tag_int)
        await self._fire_callbacks(
            "path_discovery_response",
            tag_bytes,
            contact_pubkey,
            out_len_byte,
            out_path,
            in_len_byte,
            in_path,
        )
        return True

    async def _fire_callbacks(self, event_name: str, *args: Any) -> None:
        for callback in self._push_callbacks.get(event_name, []):
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback(*args)
                else:
                    callback(*args)
            except Exception as e:
                logger.error("Error in %s callback: %s", event_name, e)

    def _spawn_background_task(self, coro: Any, label: str) -> asyncio.Task:
        """Create a fire-and-forget task that is tracked until done.

        Holding a reference prevents premature garbage collection (see the
        asyncio.create_task docs) and the done callback surfaces exceptions
        that would otherwise be silently dropped.
        """
        task = asyncio.get_running_loop().create_task(coro)
        self._background_tasks.add(task)

        def _on_done(t: asyncio.Task) -> None:
            self._background_tasks.discard(t)
            if not t.cancelled() and t.exception() is not None:
                logger.error("Background task %s failed: %s", label, t.exception())

        task.add_done_callback(_on_done)
        return task

    def _schedule_fire_callbacks(self, event_name: str, *args: Any) -> None:
        """Schedule _fire_callbacks from sync code (e.g. set_channel). No-op if no running loop."""
        try:
            self._spawn_background_task(
                self._fire_callbacks(event_name, *args), f"fire_callbacks({event_name})"
            )
        except RuntimeError:
            pass
