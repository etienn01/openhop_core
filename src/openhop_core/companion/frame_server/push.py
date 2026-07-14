"""Push frames (radio -> app, unsolicited): bridge event subscriptions and
the public push_* methods called directly by the host application."""

import asyncio
import logging
import struct
import time

from ...protocol.packet_utils import PathUtils
from ..constants import (
    MAX_PAYLOAD_SIZE,
    OUT_PATH_UNKNOWN,
    PUB_KEY_SIZE,
    PUSH_CODE_BINARY_RESPONSE,
    PUSH_CODE_CONTACT_DELETED,
    PUSH_CODE_CONTACTS_FULL,
    PUSH_CODE_CONTROL_DATA,
    PUSH_CODE_LOG_RX_DATA,
    PUSH_CODE_MSG_WAITING,
    PUSH_CODE_PATH_DISCOVERY_RESPONSE,
    PUSH_CODE_PATH_UPDATED,
    PUSH_CODE_RAW_DATA,
    PUSH_CODE_SEND_CONFIRMED,
    PUSH_CODE_TRACE_DATA,
)
from ..models import Contact
from .frames import _build_advert_push_frames

logger = logging.getLogger("CompanionFrameServer")


class _PushMixin:
    """Bridge-event push callbacks and public push_* methods of
    :class:`CompanionFrameServer`."""

    def _setup_push_callbacks(self) -> None:
        """Subscribe to bridge events and send PUSH frames to connected client."""
        # Clear any callbacks registered by a previous connection so they
        # don't accumulate across reconnections.
        self.bridge.clear_push_callbacks()
        self.bridge.on_message_received(self._on_message_received)
        self.bridge.on_channel_message_received(self._on_channel_message_received)
        self.bridge.on_channel_data_received(self._on_channel_data_received)
        self.bridge.on_send_confirmed(self._on_send_confirmed)
        self.bridge.on_advert_received(self._on_advert_received)
        self.bridge.on_node_discovered(self._on_node_discovered)
        self.bridge.on_contact_path_updated(self._on_contact_path_updated)
        self.bridge.on_binary_response(self._on_binary_response)
        self.bridge.on_path_discovery_response(self._on_path_discovery_response)
        self.bridge.on_contact_deleted(self._on_contact_deleted)
        self.bridge.on_contacts_full(self._on_contacts_full)
        self.bridge.on_raw_data_received(self._on_raw_data_received)

    # -------------------------------------------------------------------------
    # Bridge event callbacks (registered by _setup_push_callbacks)
    # -------------------------------------------------------------------------

    async def _on_message_received(
        self,
        sender_key,
        text,
        timestamp,
        txt_type,
        packet_hash=None,
        snr=None,
        rssi=None,
        sender_prefix=b"",
        path_len=0,
        queued=True,
    ):
        msg_dict = {
            "sender_key": sender_key,
            "text": text,
            "timestamp": timestamp,
            "txt_type": txt_type,
            "is_channel": False,
            "channel_idx": 0,
            "path_len": path_len,
            "packet_hash": packet_hash,
            "snr": snr,
            "rssi": rssi,
            "sender_prefix": sender_prefix,
        }
        if queued:
            await self._persist_companion_message(msg_dict)
        self._enqueue_frame(bytes([PUSH_CODE_MSG_WAITING]))

    async def _on_channel_message_received(
        self,
        channel_name,
        sender_name,
        message_text,
        timestamp,
        path_len=0,
        channel_idx=0,
        packet_hash=None,
        snr=None,
        rssi=None,
        queued=True,
    ):
        msg_dict = {
            "sender_key": b"",
            "text": message_text,
            "timestamp": timestamp,
            "txt_type": 0,
            "is_channel": True,
            "channel_idx": channel_idx,
            "path_len": path_len,
            "packet_hash": packet_hash,
            "snr": snr,
            "rssi": rssi,
        }
        if queued:
            await self._persist_companion_message(msg_dict)
        self._enqueue_frame(bytes([PUSH_CODE_MSG_WAITING]))

    async def _on_channel_data_received(
        self,
        channel_idx,
        path_len,
        data_type,
        payload,
        packet_hash=None,
        snr=None,
        rssi=None,
        queued=True,
    ):
        msg_dict = {
            "sender_key": b"",
            "text": "",
            "timestamp": 0,
            "txt_type": 0,
            "is_channel": True,
            "channel_idx": channel_idx,
            "path_len": path_len,
            "packet_hash": packet_hash,
            "snr": snr,
            "rssi": rssi,
            "channel_data_type": data_type,
            "channel_data_payload": bytes(payload or b""),
        }
        if queued:
            await self._persist_companion_message(msg_dict)
        self._enqueue_frame(bytes([PUSH_CODE_MSG_WAITING]))

    def _on_send_confirmed(self, crc, trip_ms=0):
        # Final 4 bytes are the elapsed milliseconds from send to ACK (firmware
        # processAck writes trip_time here); 0 only when the send time is unknown.
        data = struct.pack(
            "<B4sI",
            PUSH_CODE_SEND_CONFIRMED,
            struct.pack("<I", crc)[:4],
            int(trip_ms) & 0xFFFFFFFF,
        )
        self._enqueue_frame(data)

    async def _push_advert_frames(self, contact: Contact, *, is_new: bool) -> None:
        """Push the advert frame for a contact, mirroring firmware onDiscoveredContact:
        the full NEW_ADVERT when the contact was not stored (``is_new``), otherwise the
        short ADVERT (pubkey only) for a stored contact.
        """
        pubkey = contact.public_key
        if not isinstance(pubkey, bytes) or len(pubkey) < PUB_KEY_SIZE:
            return
        short, full = await asyncio.to_thread(_build_advert_push_frames, contact)
        if is_new:
            if full is not None:
                self._enqueue_frame(full)
        else:
            self._enqueue_frame(short)

    async def _on_advert_received(self, contact):
        """Stored advert: persist the contact. The client frame push is handled by
        _on_node_discovered (firmware onDiscoveredContact)."""
        try:
            if not isinstance(contact, Contact):
                logger.warning(
                    "advert_received: expected Contact, got %s — converting",
                    type(contact).__name__,
                )
                contact = Contact.from_dict(contact) if isinstance(contact, dict) else contact
            await self._maybe_persist_contact(contact)
        except Exception as e:
            logger.exception("advert_received callback error: %s", e)

    async def _on_node_discovered(self, contact_or_data):
        """Firmware onDiscoveredContact: fires for every valid advert. Pushes the full
        NEW_ADVERT when the contact was not stored, else the short ADVERT for a stored
        contact. "Stored" is determined by whether the contact is in the contact book
        (the advert pipeline has already applied any auto-add by this point)."""
        try:
            if isinstance(contact_or_data, Contact):
                contact = contact_or_data
            elif isinstance(contact_or_data, dict):
                contact = Contact.from_dict(contact_or_data, now=int(time.time()))
            else:
                return
            if not contact.name:
                return
            is_new = self.bridge.contacts.get_by_key(contact.public_key) is None
            await self._push_advert_frames(contact, is_new=is_new)
        except Exception as e:
            logger.exception("node_discovered callback error: %s", e)

    async def _on_contact_path_updated(self, contact):
        # Defense-in-depth: only push PATH and persist for known contacts
        # (mirrors firmware which does not send PATH for non-contacts).
        if not (
            hasattr(contact, "public_key")
            and isinstance(contact.public_key, bytes)
            and len(contact.public_key) >= PUB_KEY_SIZE
        ):
            return
        if not self.bridge.contacts.get_by_key(contact.public_key):
            return
        self._enqueue_frame(bytes([PUSH_CODE_PATH_UPDATED]) + contact.public_key[:PUB_KEY_SIZE])
        try:
            await self._maybe_persist_contact(contact)
        except Exception as e:
            logger.warning("Persist contact after path update failed: %s", e)

    def _on_binary_response(self, tag_bytes, response_data, parsed=None, request_type=None):
        if isinstance(tag_bytes, bytes):
            if len(tag_bytes) < 4:
                return
            tag_int = struct.unpack("<I", tag_bytes[:4])[0]
        else:
            try:
                tag_int = int(tag_bytes)
            except (TypeError, ValueError):
                return
        if tag_int not in self._companion_binary_tags:
            # Region discovery replies can be generated outside this
            # frame-server request path; allow those through for parity
            # with pre-owner-tag behavior while keeping other responses
            # isolated to the requesting virtual companion.
            if not (isinstance(parsed, dict) and parsed.get("type") == "regions"):
                return
        else:
            self._companion_binary_tags.discard(tag_int)
        frame = (
            bytes([PUSH_CODE_BINARY_RESPONSE, 0])
            + (tag_bytes if isinstance(tag_bytes, bytes) else struct.pack("<I", tag_bytes))
            + response_data
        )
        self._enqueue_frame(frame)

    def _on_path_discovery_response(self, tag_bytes, contact_pubkey, out_path, in_path):
        pub_key_prefix = (
            contact_pubkey if isinstance(contact_pubkey, bytes) else bytes.fromhex(contact_pubkey)
        )[:6]
        out_path = out_path if isinstance(out_path, bytes) else bytes(out_path)
        in_path = in_path if isinstance(in_path, bytes) else bytes(in_path)
        frame = (
            bytes([PUSH_CODE_PATH_DISCOVERY_RESPONSE, 0])
            + pub_key_prefix
            + bytes([len(out_path)])
            + out_path
            + bytes([len(in_path)])
            + in_path
        )
        self._enqueue_frame(frame)

    def _on_contact_deleted(self, pub_key):
        if isinstance(pub_key, bytes) and len(pub_key) >= PUB_KEY_SIZE:
            self._enqueue_frame(bytes([PUSH_CODE_CONTACT_DELETED]) + pub_key[:PUB_KEY_SIZE])

    def _on_contacts_full(self):
        self._enqueue_frame(bytes([PUSH_CODE_CONTACTS_FULL]))

    def _on_raw_data_received(self, payload_bytes: bytes, snr: float, rssi: int) -> None:
        """Push PUSH_CODE_RAW_DATA (0x84): code, SNR byte, RSSI byte,
        path-len byte (unknown), payload."""
        snr_byte = max(-128, min(127, int(round(snr * 4))))
        rssi_byte = max(-128, min(127, int(rssi)))
        payload_len = min(len(payload_bytes), MAX_PAYLOAD_SIZE - 4)
        data = (
            bytes([PUSH_CODE_RAW_DATA])
            + struct.pack("<bb", snr_byte, rssi_byte)
            + bytes([OUT_PATH_UNKNOWN])
            + payload_bytes[:payload_len]
        )
        self._enqueue_frame(data)

    # -------------------------------------------------------------------------
    # Public push methods (called directly by host application)
    # -------------------------------------------------------------------------

    def push_trace_data(
        self,
        path_len: int,
        flags: int,
        tag: int,
        auth_code: int,
        path_hashes: bytes,
        path_snrs: bytes,
        final_snr_byte: int,
    ) -> None:
        """Push PUSH_CODE_TRACE_DATA (0x89) to client.  Matches firmware
        ``onTraceRecv()`` frame format.

        Sync, non-blocking.  Safe to call from any context (async or sync),
        like :pymethod:`push_rx_raw`.
        """
        if self._write_queue is None:
            return
        hash_width = PathUtils.trace_payload_hash_width(flags)
        expected_snr_len = path_len // hash_width
        if len(path_snrs) != expected_snr_len:
            logger.debug(
                "push_trace_data: path_snrs len %s != expected %s",
                len(path_snrs),
                expected_snr_len,
            )
            return
        data = (
            bytes([PUSH_CODE_TRACE_DATA, 0, path_len, flags])
            + struct.pack("<II", tag & 0xFFFFFFFF, auth_code & 0xFFFFFFFF)
            + path_hashes
            + path_snrs
            + bytes([final_snr_byte & 0xFF])
        )
        self._enqueue_frame(data)

    async def push_trace_data_async(
        self,
        path_len: int,
        flags: int,
        tag: int,
        auth_code: int,
        path_hashes: bytes,
        path_snrs: bytes,
        final_snr_byte: int,
    ) -> None:
        """Async wrapper for code that must ``await`` a coroutine (same as
        :pymethod:`push_trace_data`).
        """
        self.push_trace_data(
            path_len,
            flags,
            tag,
            auth_code,
            path_hashes,
            path_snrs,
            final_snr_byte,
        )

    def push_rx_raw(self, snr: float, rssi: int, raw: bytes) -> None:
        """Push raw RX packet to client (PUSH_CODE_LOG_RX_DATA 0x88).

        Sync, non-blocking.  Safe to call from any context (async or sync).
        """
        if self._write_queue is None:
            return
        snr_byte = max(-128, min(127, int(round(snr * 4))))
        rssi_byte = max(-128, min(127, int(rssi)))
        if snr_byte < 0:
            snr_byte += 256
        if rssi_byte < 0:
            rssi_byte += 256
        payload_len = min(len(raw), MAX_PAYLOAD_SIZE - 3)  # 3 = code + snr + rssi
        data = bytes([PUSH_CODE_LOG_RX_DATA, snr_byte & 0xFF, rssi_byte & 0xFF]) + raw[:payload_len]
        self._enqueue_frame(data)

    async def push_rx_raw_async(self, snr: float, rssi: int, raw: bytes) -> None:
        """Push raw RX packet to client.  Async wrapper for backward compatibility."""
        self.push_rx_raw(snr, rssi, raw)

    async def push_control_data(
        self,
        snr: float,
        rssi: int,
        path_len: int,
        path_bytes: bytes,
        payload: bytes,
    ) -> None:
        """Push CONTROL packet to client (PUSH_CODE_CONTROL_DATA 0x8E).

        Kept as ``async def`` for backward-compatible call sites that
        ``await`` it, but the body is synchronous (just enqueues).
        """
        if self._write_queue is None:
            logger.warning("Push control data skipped: no client connection")
            return
        # Discovery response (0x90): clear only no-op callbacks owned by this
        # frame server (created by CMD_SEND_CONTROL_DATA discovery requests).
        if self._control_handler and len(payload) >= 6 and (payload[0] & 0xF0) == 0x90:
            tag = struct.unpack("<I", payload[2:6])[0]
            if tag in self._companion_discovery_tags:
                self._control_handler.clear_response_callback(tag)
                self._companion_discovery_tags.discard(tag)
        snr_val = snr if isinstance(snr, (int, float)) else 0.0
        rssi_val = rssi if isinstance(rssi, (int, float)) else 0
        snr_byte = max(-128, min(127, int(round(float(snr_val) * 4))))
        rssi_byte = max(-128, min(127, int(rssi_val)))
        if snr_byte < 0:
            snr_byte += 256
        if rssi_byte < 0:
            rssi_byte += 256
        path_len_byte = max(0, min(255, int(path_len) if path_len is not None else 0))
        payload_max = MAX_PAYLOAD_SIZE - 4  # 4 = code + snr + rssi + path_len_byte
        payload_slice = bytes(payload[:payload_max]) if payload else b""
        data = (
            bytes(
                [
                    PUSH_CODE_CONTROL_DATA,
                    snr_byte & 0xFF,
                    rssi_byte & 0xFF,
                    path_len_byte,
                ]
            )
            + payload_slice
        )
        self._enqueue_frame(data)
        logger.debug("Pushed control data 0x8E to client: payload_len=%s", len(payload_slice))
