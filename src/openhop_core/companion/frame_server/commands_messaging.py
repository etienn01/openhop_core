"""Messaging command handlers: text/channel sends, sync-next-message,
login/status/telemetry, binary/anon/control/trace/raw requests."""

import asyncio
import logging
import struct

from ...protocol.constants import TELEM_PERM_BASE, TELEM_PERM_ENVIRONMENT, TELEM_PERM_LOCATION
from ...protocol.packet_utils import PathUtils
from ..constants import (
    ERR_CODE_BAD_STATE,
    ERR_CODE_ILLEGAL_ARG,
    ERR_CODE_NOT_FOUND,
    ERR_CODE_TABLE_FULL,
    ERR_CODE_UNSUPPORTED_CMD,
    FIRMWARE_VER_CODE,
    LOGIN_TIMEOUT_HINT_MS,
    MAX_CHANNEL_DATA_LENGTH,
    MAX_PATH_SIZE,
    OUT_PATH_UNKNOWN,
    PUB_KEY_SIZE,
    PUSH_CODE_LOGIN_FAIL,
    PUSH_CODE_LOGIN_SUCCESS,
    PUSH_CODE_STATUS_RESPONSE,
    PUSH_CODE_TELEMETRY_RESPONSE,
    RESP_CODE_CHANNEL_DATA_RECV,
    RESP_CODE_CHANNEL_MSG_RECV,
    RESP_CODE_CHANNEL_MSG_RECV_V3,
    RESP_CODE_CONTACT_MSG_RECV,
    RESP_CODE_CONTACT_MSG_RECV_V3,
    RESP_CODE_NO_MORE_MESSAGES,
    STATUS_TIMEOUT_HINT_MS,
    TELEMETRY_TIMEOUT_HINT_MS,
    TRACE_BASE_TIMEOUT_MS,
    TRACE_PER_PATH_BYTE_TIMEOUT_MS,
    TXT_MSG_TIMEOUT_HINT_MS,
    TXT_TYPE_CLI_DATA,
    TXT_TYPE_SIGNED_PLAIN,
)
from ..models import QueuedMessage

logger = logging.getLogger("CompanionFrameServer")


class _MessagingCommandsMixin:
    """Messaging and request _cmd_* handlers of :class:`CompanionFrameServer`."""

    async def _cmd_send_txt_msg(self, data: bytes) -> None:
        if len(data) < 12:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        txt_type = data[0]
        attempt = data[1]
        # data[2:6] = host-supplied msg_timestamp (LE uint32). Used as-is for plain DMs so
        # retries of the same message share a stable timestamp (mirrors firmware sendMessage).
        # For CLI_DATA — or when the host omits it (0) — mint a fresh timestamp instead,
        # matching firmware which overrides CLI_DATA with the RTC to avoid replay protection.
        host_timestamp = int.from_bytes(data[2:6], "little")
        use_timestamp = (
            None if (txt_type == TXT_TYPE_CLI_DATA or host_timestamp == 0) else host_timestamp
        )
        pubkey_prefix = data[6:12]
        text = data[12:].decode("utf-8", errors="replace").rstrip("\x00")
        contact = self.bridge.contacts.get_by_key_prefix(pubkey_prefix)
        if not contact:
            self._write_err(ERR_CODE_NOT_FOUND)
            return
        result = await self.bridge.send_text_message(
            contact.public_key_bytes,
            text,
            txt_type=txt_type,
            attempt=attempt,
            wait_for_ack=False,
            timestamp=use_timestamp,
        )
        if result.success:
            self._write_sent_result(result, default_timeout_ms=TXT_MSG_TIMEOUT_HINT_MS)
        else:
            self._write_err(ERR_CODE_BAD_STATE)

    async def _cmd_send_channel_txt_msg(self, data: bytes) -> None:
        if len(data) < 6:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        txt_type = data[0]
        channel_idx = data[1]
        text = data[6:].decode("utf-8", errors="replace").rstrip("\x00")
        if txt_type != 0:
            self._write_err(ERR_CODE_UNSUPPORTED_CMD)
            return
        if self.bridge.get_channel(channel_idx) is None:
            self._write_err(ERR_CODE_NOT_FOUND)
            return
        ok = await self.bridge.send_channel_message(channel_idx, text)
        self._write_ok() if ok else self._write_err(ERR_CODE_BAD_STATE)

    async def _cmd_send_channel_data(self, data: bytes) -> None:
        """Handle CMD_SEND_CHANNEL_DATA (62)."""
        if len(data) < 4:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        channel_idx = data[0]
        path_len = data[1]
        if self.bridge.get_channel(channel_idx) is None:
            self._write_err(ERR_CODE_NOT_FOUND)
            return
        offset = 2
        path = b""
        if path_len != OUT_PATH_UNKNOWN:
            if not PathUtils.is_valid_path_len(path_len):
                self._write_err(ERR_CODE_ILLEGAL_ARG)
                return
            path_byte_len = PathUtils.get_path_byte_len(path_len)
            if len(data) < offset + path_byte_len + 2:
                self._write_err(ERR_CODE_ILLEGAL_ARG)
                return
            path = data[offset : offset + path_byte_len]
            offset += path_byte_len
        if len(data) < offset + 2:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        data_type = int.from_bytes(data[offset : offset + 2], "little")
        payload = data[offset + 2 :]
        if data_type == 0 or len(payload) > MAX_CHANNEL_DATA_LENGTH:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        send_channel_data = getattr(self.bridge, "send_channel_data", None)
        if not send_channel_data:
            self._write_err(ERR_CODE_UNSUPPORTED_CMD)
            return
        ok = await send_channel_data(
            channel_idx,
            data_type,
            payload,
            path=path if path_len != OUT_PATH_UNKNOWN else None,
            path_len_encoded=path_len,
        )
        if ok:
            self._write_ok()
        else:
            self._write_err(ERR_CODE_TABLE_FULL)

    async def _cmd_send_binary_req(self, data: bytes) -> None:
        if len(data) < PUB_KEY_SIZE + 1:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        pubkey = data[:PUB_KEY_SIZE]
        req_data = data[PUB_KEY_SIZE:]
        send_binary_req = getattr(self.bridge, "send_binary_req", None)
        if not send_binary_req:
            self._write_err(ERR_CODE_UNSUPPORTED_CMD)
            return
        try:
            result = await send_binary_req(pubkey, req_data)
        except Exception as e:
            logger.error("send_binary_req error: %s", e, exc_info=True)
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        if not result.success:
            self._write_err(ERR_CODE_NOT_FOUND)
            return
        self._write_sent_result(result, own_binary_tag=True)

    async def _cmd_send_anon_req(self, data: bytes) -> None:
        if len(data) < PUB_KEY_SIZE + 1:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        pubkey = data[:PUB_KEY_SIZE]
        req_data = data[PUB_KEY_SIZE:]
        send_anon_req = getattr(self.bridge, "send_anon_req", None)
        if not send_anon_req:
            self._write_err(ERR_CODE_UNSUPPORTED_CMD)
            return
        try:
            result = await send_anon_req(pubkey, req_data)
        except Exception as e:
            logger.error("send_anon_req error: %s", e, exc_info=True)
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        if not result.success:
            # FW PR #2672: anon req no longer returns NOT_FOUND. Both "couldn't add
            # transient contact" and "send failed" map to ERR_CODE_TABLE_FULL.
            self._write_err(ERR_CODE_TABLE_FULL)
            return
        self._write_sent_result(result, own_binary_tag=True)

    async def _cmd_send_control_data(self, data: bytes) -> None:
        if len(data) < 2:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        if (data[0] & 0x80) == 0:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        # Discovery request: register a no-op response callback
        if self._control_handler and len(data) >= 6 and (data[0] & 0xF0) == 0x80:
            tag = struct.unpack("<I", data[2:6])[0]
            self._companion_discovery_tags.add(tag)
            self._control_handler.set_response_callback(tag, lambda _: None)
        send_control = getattr(self.bridge, "send_control_data", None)
        if not send_control:
            self._write_err(ERR_CODE_UNSUPPORTED_CMD)
            return
        try:
            ok = await send_control(data)
        except Exception as e:
            logger.error("send_control_data error: %s", e, exc_info=True)
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        if ok:
            self._write_ok()
        else:
            self._write_err(ERR_CODE_TABLE_FULL)

    async def _cmd_send_path_discovery_req(self, data: bytes) -> None:
        logger.info(
            "Path discovery request received (cmd 52), data_len=%s",
            len(data),
        )
        if len(data) < 1 + PUB_KEY_SIZE:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        pub_key = data[1 : 1 + PUB_KEY_SIZE]
        send_req = getattr(self.bridge, "send_path_discovery_req", None)
        if not send_req:
            self._write_err(ERR_CODE_UNSUPPORTED_CMD)
            return
        try:
            result = await send_req(pub_key)
        except Exception as e:
            logger.error("send_path_discovery_req error: %s", e, exc_info=True)
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        if not result.success:
            self._write_err(ERR_CODE_NOT_FOUND)
            return
        self._write_sent_result(result)

    async def _cmd_send_trace_path(self, data: bytes) -> None:
        if len(data) < 10:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        tag = struct.unpack_from("<I", data, 0)[0]
        auth_code = struct.unpack_from("<I", data, 4)[0]
        flags = data[8]
        path_bytes = data[9:]
        path_len = len(path_bytes)
        hash_width = PathUtils.trace_payload_hash_width(flags)
        if (path_len // hash_width) > MAX_PATH_SIZE or (path_len % hash_width) != 0:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        send_raw = getattr(self.bridge, "send_trace_path_raw", None)
        if not send_raw:
            self._write_err(ERR_CODE_UNSUPPORTED_CMD)
            return
        try:
            ok = await send_raw(tag, auth_code, flags, path_bytes)
        except Exception as e:
            logger.error("send_trace_path error: %s", e, exc_info=True)
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        if not ok:
            self._write_err(ERR_CODE_TABLE_FULL)
            return
        est_timeout_ms = TRACE_BASE_TIMEOUT_MS + (path_len * TRACE_PER_PATH_BYTE_TIMEOUT_MS)
        self._write_sent_response(False, tag, est_timeout_ms)
        # If we are the final hop, push trace data immediately
        if path_bytes and self.local_hash is not None and path_bytes[-1] == self.local_hash:
            snr_len = path_len // hash_width
            path_snrs = bytes(snr_len)
            final_snr_byte = 0
            self.push_trace_data(
                path_len,
                flags,
                tag,
                auth_code,
                path_bytes,
                path_snrs,
                final_snr_byte,
            )

    def _build_message_frame(self, msg: "QueuedMessage") -> bytes:
        """Encode a QueuedMessage into a response frame (shared by base and subclasses)."""
        snr_byte = max(-128, min(127, int(round(getattr(msg, "snr", 0) * 4))))
        if snr_byte < 0:
            snr_byte += 256
        if msg.is_channel:
            path_len_byte = msg.path_len if msg.path_len < 256 else 0xFF
            if getattr(msg, "channel_data_type", 0):
                payload = bytes(getattr(msg, "channel_data_payload", b"") or b"")
                payload = payload[:MAX_CHANNEL_DATA_LENGTH]
                return (
                    bytes(
                        [
                            RESP_CODE_CHANNEL_DATA_RECV,
                            snr_byte & 0xFF,
                            0,
                            0,
                            msg.channel_idx,
                            path_len_byte,
                        ]
                    )
                    + struct.pack("<H", msg.channel_data_type & 0xFFFF)
                    + bytes([len(payload)])
                    + payload
                )
            txt_type = 0
            text_bytes = (msg.text or "").rstrip("\x00").encode("utf-8", errors="replace")
            if self._app_target_ver >= 3:
                return (
                    bytes(
                        [
                            RESP_CODE_CHANNEL_MSG_RECV_V3,
                            snr_byte & 0xFF,
                            0,
                            0,
                            msg.channel_idx,
                            path_len_byte,
                            txt_type,
                        ]
                    )
                    + struct.pack("<I", msg.timestamp)
                    + text_bytes
                )
            return (
                bytes(
                    [
                        RESP_CODE_CHANNEL_MSG_RECV,
                        msg.channel_idx,
                        path_len_byte,
                        txt_type,
                    ]
                )
                + struct.pack("<I", msg.timestamp)
                + text_bytes
            )
        prefix = (
            msg.sender_key[:6] if len(msg.sender_key) >= 6 else msg.sender_key.ljust(6, b"\x00")
        )
        path_len_byte = msg.path_len if msg.path_len < 256 else 0xFF
        text_bytes = msg.text.encode("utf-8", errors="replace")
        extra = b""
        if msg.txt_type == TXT_TYPE_SIGNED_PLAIN:
            # Firmware queueMessage() inserts the 4-byte author pubkey prefix
            # between the timestamp and the text for signed (room server)
            # messages; the app consumes these 4 bytes to attribute the author.
            author = bytes(getattr(msg, "sender_prefix", b"") or b"")
            extra = author[:4].ljust(4, b"\x00")
        if self._app_target_ver >= 3:
            return (
                bytes([RESP_CODE_CONTACT_MSG_RECV_V3, snr_byte & 0xFF, 0, 0])
                + prefix
                + bytes([path_len_byte, msg.txt_type])
                + struct.pack("<I", msg.timestamp)
                + extra
                + text_bytes
            )
        return (
            bytes([RESP_CODE_CONTACT_MSG_RECV])
            + prefix
            + bytes([path_len_byte, msg.txt_type])
            + struct.pack("<I", msg.timestamp)
            + extra
            + text_bytes
        )

    async def _cmd_sync_next_message(self, data: bytes) -> None:
        msg = self.bridge.sync_next_message()
        if msg is None:
            msg = await asyncio.to_thread(self._sync_next_from_persistence)
        if msg is None:
            self._write_frame(bytes([RESP_CODE_NO_MORE_MESSAGES]))
            return
        self._write_frame(self._build_message_frame(msg))

    async def _cmd_send_login(self, data: bytes) -> None:
        if len(data) < PUB_KEY_SIZE:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        pubkey = data[:PUB_KEY_SIZE]
        password = (
            data[PUB_KEY_SIZE:].decode("utf-8", errors="replace").rstrip("\x00")
            if len(data) > PUB_KEY_SIZE
            else ""
        )
        self._write_sent_response(True, 0, LOGIN_TIMEOUT_HINT_MS)
        result = await self.bridge.send_login(pubkey, password)
        if result.get("success"):
            # Layout matches MeshCore companion_radio onContactResponse
            fw_level = result.get("firmware_ver_level")
            if fw_level is None:
                fw_level = FIRMWARE_VER_CODE  # fallback so app sees >= 2 for owner info
            self._write_frame(
                bytes(
                    [
                        PUSH_CODE_LOGIN_SUCCESS,
                        1 if result.get("is_admin") else 0,
                    ]
                )
                + pubkey[:6]
                + struct.pack("<I", result.get("tag", 0))
                + bytes([result.get("acl_permissions", 0)])
                + bytes([min(255, max(0, int(fw_level)))])
            )
        else:
            self._write_frame(bytes([PUSH_CODE_LOGIN_FAIL, 0]) + pubkey[:6])

    async def _cmd_send_status_req(self, data: bytes) -> None:
        if len(data) < PUB_KEY_SIZE:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        pubkey = data[0:PUB_KEY_SIZE]
        self._write_sent_response(False, 0, STATUS_TIMEOUT_HINT_MS)
        result = await self.bridge.send_status_request(pubkey)
        if not result.get("success"):
            logger.debug("Status request failed for %s; no push sent)", pubkey[:6].hex())
            return
        stats_data = result.get("stats", {})
        raw_bytes = stats_data.get("raw_bytes", b"")
        if not raw_bytes:
            logger.debug(
                "Status response had no raw_bytes for %s; no push sent",
                pubkey[:6].hex(),
            )
            return
        self._write_frame(bytes([PUSH_CODE_STATUS_RESPONSE, 0]) + pubkey[:6] + raw_bytes)

    async def _cmd_send_telemetry_req(self, data: bytes) -> None:
        # Protocol: CMD_SEND_TELEMETRY_REQ has reserved bytes(3) then pub_key bytes(32).
        # See MeshCore Companion-Radio-Protocol: CMD_SEND_TELEMETRY_REQ frame format.
        if len(data) < 3 + PUB_KEY_SIZE:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        pubkey = data[3 : 3 + PUB_KEY_SIZE]
        # Request all: base + location + environment
        flags = TELEM_PERM_BASE | TELEM_PERM_LOCATION | TELEM_PERM_ENVIRONMENT
        want_base = bool(flags & TELEM_PERM_BASE)
        want_location = bool(flags & TELEM_PERM_LOCATION)
        want_environment = bool(flags & TELEM_PERM_ENVIRONMENT)
        self._write_sent_response(False, 0, TELEMETRY_TIMEOUT_HINT_MS)
        result = await self.bridge.send_telemetry_request(
            pubkey,
            want_base=want_base,
            want_location=want_location,
            want_environment=want_environment,
        )
        if not result.get("success"):
            logger.debug("Telemetry request failed for %s; no push sent", pubkey[:6].hex())
            return
        telem_data = result.get("telemetry_data", {})
        raw_bytes = telem_data.get("raw_bytes", b"")
        if not raw_bytes:
            logger.debug(
                "Telemetry response had no raw_bytes for %s; no push sent",
                pubkey[:6].hex(),
            )
            return
        self._write_frame(bytes([PUSH_CODE_TELEMETRY_RESPONSE, 0]) + pubkey[:6] + raw_bytes)
        logger.info("Telemetry push sent to client: %d bytes LPP", len(raw_bytes))

    async def _cmd_logout(self, data: bytes) -> None:
        if len(data) < PUB_KEY_SIZE:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        pubkey = data[:PUB_KEY_SIZE]
        await self.bridge.send_logout(pubkey)
        self._write_ok()

    async def _cmd_send_raw_data(self, data: bytes) -> None:
        """Handle CMD_SEND_RAW_DATA (25).
        Format: [path_len_encoded][path][payload] (min 4-byte payload)."""
        if len(data) < 6:
            self._write_err(ERR_CODE_UNSUPPORTED_CMD)
            return
        path_len_byte = data[0]
        if not PathUtils.is_valid_path_len(path_len_byte):
            self._write_err(ERR_CODE_UNSUPPORTED_CMD)
            return
        path_byte_len = PathUtils.get_path_byte_len(path_len_byte)
        if 1 + path_byte_len + 4 > len(data):
            self._write_err(ERR_CODE_UNSUPPORTED_CMD)
            return
        path = data[1 : 1 + path_byte_len]
        payload = data[1 + path_byte_len :]
        result = await self.bridge.send_raw_data_direct(
            path, payload, path_len_encoded=path_len_byte
        )
        if result.success:
            self._write_ok()
        else:
            self._write_err(ERR_CODE_TABLE_FULL)

    async def _cmd_send_raw_packet(self, data: bytes) -> None:
        """Handle CMD_SEND_RAW_PACKET (65). Format: [priority(1)][raw_packet...].

        Mirrors MyMesh.cpp:1967: inject a low-level packet with a TX priority.
        Delegates to the bridge's ``send_raw_packet`` if available.
        """
        if len(data) < 3:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        priority = data[0]
        packet_bytes = data[1:]
        send_raw_packet = getattr(self.bridge, "send_raw_packet", None)
        if not send_raw_packet:
            self._write_err(ERR_CODE_UNSUPPORTED_CMD)
            return
        try:
            ok = await send_raw_packet(priority, packet_bytes)
        except Exception as e:
            logger.error("send_raw_packet error: %s", e, exc_info=True)
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        if ok:
            self._write_ok()
        else:
            self._write_err(ERR_CODE_TABLE_FULL)
