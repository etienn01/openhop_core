"""Unified TX/send operations of CompanionBase (Radio and Bridge)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import struct
import time
from typing import Any, Awaitable, Callable, Optional

from ..protocol import Packet, PacketBuilder
from ..protocol.constants import (
    ADVERT_FLAG_HAS_LOCATION,
    ADVERT_FLAG_HAS_NAME,
    MAX_PACKET_PAYLOAD,
    MAX_PATH_SIZE,
    PAYLOAD_TYPE_ADVERT,
    PAYLOAD_TYPE_CONTROL,
    PAYLOAD_TYPE_GRP_DATA,
    PH_ROUTE_MASK,
    PUB_KEY_SIZE,
    REQ_TYPE_GET_STATUS,
    REQ_TYPE_GET_TELEMETRY_DATA,
    ROUTE_TYPE_DIRECT,
    TELEM_PERM_BASE,
)
from .base_support import ResponseWaiter, _fmt_path, adv_type_to_flags
from .constants import (
    ADV_TYPE_NONE,
    ADVERT_LOC_SHARE,
    DEFAULT_RESPONSE_TIMEOUT_MS,
    MAX_PENDING_ACK_CRCS,
    PROTOCOL_CODE_ANON_REQ,
    PROTOCOL_CODE_BINARY_REQ,
    PROTOCOL_CODE_RAW_DATA,
    PUSH_CODE_TELEMETRY_RESPONSE,
    TXT_TYPE_CLI_DATA,
    TXT_TYPE_PLAIN,
)
from .models import Contact, QueuedMessage, SentResult
from .timing import DEFAULT_MAX_ATTEMPTS, response_timeout_ms

logger = logging.getLogger("CompanionBase")


class _SendOpsMixin:
    """Part of :class:`CompanionBase` (see companion_base.py)."""

    # -------------------------------------------------------------------------
    # Unified TX methods (shared between Radio and Bridge)
    # -------------------------------------------------------------------------

    async def advertise(self, flood: bool = True) -> bool:
        """Broadcast an advertisement packet."""
        flags = adv_type_to_flags(self.prefs.adv_type)
        flags |= ADVERT_FLAG_HAS_NAME
        lat, lon = 0.0, 0.0
        if self.prefs.advert_loc_policy == ADVERT_LOC_SHARE:
            lat, lon = self.prefs.latitude, self.prefs.longitude
            if lat != 0.0 or lon != 0.0:
                flags |= ADVERT_FLAG_HAS_LOCATION
        route = "flood" if flood else "direct"
        pkt = PacketBuilder.create_advert(
            local_identity=self._identity,
            name=self.prefs.node_name,
            lat=lat,
            lon=lon,
            flags=flags,
            route_type=route,
        )
        self._apply_flood_scope(pkt)
        self._apply_path_hash_mode(pkt)
        success = await self._send_packet(pkt, wait_for_ack=False)
        if success:
            self.stats.record_tx(is_flood=flood)
        else:
            self.stats.record_tx_error()
        return success

    async def share_contact(self, pub_key: bytes) -> bool:
        """Share a contact's advert on zero hops (direct route, empty path).

        Matches firmware ``BaseChatMesh::shareContactZeroHop``: replay the last stored
        raw ADVERT wire bytes for this contact (see ``Contact.last_advert_packet``),
        with ``Mesh::sendZeroHop``-style header/path normalization. Does not re-sign with
        the companion identity. If no blob is stored (never heard an advert for this
        contact), returns ``False``.
        """
        contact = self.contacts.get_by_key(pub_key)
        if not contact:
            return False
        blob = contact.last_advert_packet
        if not blob:
            return False
        try:
            pkt = Packet()
            if not pkt.read_from(bytes(blob)):
                return False
            if pkt.get_payload_type() != PAYLOAD_TYPE_ADVERT:
                return False
            if len(pkt.payload) >= PUB_KEY_SIZE:
                embedded = bytes(pkt.payload[:PUB_KEY_SIZE])
                if embedded != pub_key:
                    logger.warning(
                        "Cached advert pubkey does not match contact key; refusing share"
                    )
                    return False
            # Mesh::sendZeroHop (non-transport): direct route, path_len=0, empty path
            pkt.header = (pkt.header & ~PH_ROUTE_MASK) | ROUTE_TYPE_DIRECT
            pkt.transport_codes = [0, 0]
            pkt.path_len = 0
            pkt.path = bytearray()
            return await self._send_packet(pkt, wait_for_ack=False)
        except Exception as e:
            logger.error("Error sharing contact: %s", e)
            return False

    async def send_trace_path_raw(
        self,
        tag: int,
        auth_code: int,
        flags: int,
        path_bytes: bytes,
    ) -> bool:
        """Send a trace packet with an explicit path."""
        try:
            path_list = list(path_bytes)
            pkt = PacketBuilder.create_trace(tag, auth_code, flags, path=path_list)
            self._apply_flood_scope(pkt)
            self._apply_path_hash_mode(pkt)
            return await self._send_packet(pkt, wait_for_ack=False)
        except Exception as e:
            logger.error("Error sending trace (raw path): %s", e)
            return False

    async def send_binary_req(
        self, pub_key: bytes, data: bytes, timeout_seconds: float = 15.0
    ) -> SentResult:
        """Send binary request (CMD_SEND_BINARY_REQ).

        data = request_type(1) + optional payload.
        Returns SentResult with expected_ack (4-byte tag as int) and timeout_ms.
        """
        contact = self.contacts.get_by_key(pub_key)
        if not contact:
            return SentResult(success=False)
        # Resolve by exact public key, not name: two contacts can share a name
        # (e.g. a re-keyed node) and get_by_name returns the first match, which
        # would encrypt/route to the wrong key.
        proxy = self.contacts.get_proxy_by_key(pub_key)
        if not proxy:
            return SentResult(success=False)
        request_type = data[0] if len(data) >= 1 else 0
        # C++ companion pattern (BaseChatMesh::sendRequest):
        #   tag = getRTCClock()->getCurrentTimeUnique()
        #   memcpy(temp, &tag, 4);  memcpy(&temp[4], req_data, data_len);
        # create_protocol_request packs: timestamp(4) + protocol_code(1) + extra_data.
        # The repeater echoes sender_timestamp (bytes 0-3) in the response.
        # So the timestamp IS the tag — we capture it from create_protocol_request.
        protocol_code = request_type
        req_payload = data[1:]  # request params only; timestamp provides uniqueness
        self.cleanup_expired_binary_requests()
        try:
            pkt, timestamp = PacketBuilder.create_protocol_request(
                contact=proxy,
                local_identity=self._identity,
                protocol_code=protocol_code,
                data=req_payload,
            )
            # Use the timestamp as the tag — matches what the repeater echoes back
            tag_int = timestamp
            tag_bytes = tag_int.to_bytes(4, "little")
            tag_hex = tag_bytes.hex()
            self.register_binary_request(
                tag_hex,
                request_type=request_type,
                timeout_seconds=timeout_seconds,
                pubkey_prefix=pub_key[:6].hex(),
            )
            self._apply_flood_scope(pkt)
            self._apply_path_hash_mode(pkt)
            success = await self._send_packet(pkt, wait_for_ack=False)
        except Exception as e:
            logger.error("Binary request send error: %s", e)
            if "tag_hex" in locals():
                self._pending_binary_requests.pop(tag_hex, None)
            return SentResult(success=False)
        if not success:
            self._pending_binary_requests.pop(tag_hex, None)
            return SentResult(success=False)
        return SentResult(
            success=True,
            is_flood=contact.out_path_len <= 0,
            expected_ack=tag_int,
            timeout_ms=DEFAULT_RESPONSE_TIMEOUT_MS,
        )

    async def send_anon_req(
        self, pub_key: bytes, data: bytes, timeout_seconds: float = 15.0
    ) -> SentResult:
        """Send anonymous request (CMD_SEND_ANON_REQ), e.g. owner info.

        data = request payload (e.g. [0x07] for GET_OWNER_INFO). Response is
        delivered via on_binary_response (PUSH_CODE_BINARY_RESPONSE) like binary req.
        """
        contact = self.contacts.get_by_key(pub_key)
        if not contact:
            # FIRMWARE_VER_CODE 13+ (PR #2672): allow non-contact anon requests by
            # creating a transient zero-hop contact. Mirrors firmware sendAnonReq:
            # out_path_len=0 => direct zero-hop, type=ADV_TYPE_NONE (unknown).
            contact = Contact(
                public_key=pub_key,
                name="",
                adv_type=ADV_TYPE_NONE,
                out_path_len=0,
                out_path=b"",
                lastmod=int(time.time()),
            )
            if not self.contacts.add_transient(contact):
                return SentResult(success=False)
        # Resolve the proxy by key (anon contacts have an empty name, which
        # get_by_name would mis-match against any other empty-named contact).
        proxy = self.contacts.get_proxy_by_key(pub_key)
        if not proxy:
            return SentResult(success=False)
        request_type = PROTOCOL_CODE_ANON_REQ
        req_payload = data  # no random tag; timestamp provides uniqueness
        # The first byte is the ANON_REQ_TYPE_* sub-type (e.g. REGIONS/OWNER);
        # record it so the response can be parsed by sub-type rather than being
        # mistaken for a binary REQ_TYPE_GET_OWNER_INFO (both use code 0x07).
        anon_sub_type = req_payload[0] if len(req_payload) >= 1 else None
        self.cleanup_expired_binary_requests()
        try:
            pkt, timestamp = PacketBuilder.create_anon_request(
                contact=proxy,
                local_identity=self._identity,
                req_data=req_payload,
            )
            # Use the timestamp as the tag — matches what the repeater echoes back
            tag_int = timestamp
            tag_bytes = tag_int.to_bytes(4, "little")
            tag_hex = tag_bytes.hex()
            self._apply_flood_scope(pkt)
            self._apply_path_hash_mode(pkt)
            # Adaptive timeout (firmware calcFlood/DirectTimeoutMillisFor). This is
            # fire-and-forget: the response arrives async via the binary-response
            # push, and the client retries on this timeout hint — the same model
            # firmware uses for anon/discovery (it returns est_timeout and the host
            # app re-issues). A short adaptive hint => fast client-driven retry.
            timeout_s = self._response_timeout_s(pkt, proxy)
            self.register_binary_request(
                tag_hex,
                request_type=request_type,
                timeout_seconds=max(timeout_seconds, timeout_s * DEFAULT_MAX_ATTEMPTS),
                pubkey_prefix=pub_key[:6].hex(),
                context={"anon_sub_type": anon_sub_type},
            )
            success = await self._send_packet(pkt, wait_for_ack=False)
        except Exception as e:
            logger.error("Anon request send error: %s", e)
            if "tag_hex" in locals():
                self._pending_binary_requests.pop(tag_hex, None)
            return SentResult(success=False)
        if not success:
            self._pending_binary_requests.pop(tag_hex, None)
            return SentResult(success=False)
        return SentResult(
            success=True,
            # Direct (incl. zero-hop, out_path_len == 0) when the path is known;
            # flood only when the out_path is unknown (-1). Mirrors create_anon_request.
            is_flood=contact.out_path_len < 0,
            expected_ack=tag_int,
            timeout_ms=int(timeout_s * 1000),
        )

    async def send_path_discovery(self, pub_key: bytes) -> bool:
        """Legacy: send path discovery without returning tag. Prefer send_path_discovery_req."""
        result = await self.send_path_discovery_req(pub_key)
        return result.success

    async def send_path_discovery_req(self, pub_key: bytes) -> SentResult:
        """Send path discovery (flood telemetry request with tag).

        Returns SentResult for RESP_CODE_SENT. When path return arrives with
        matching tag, path_discovery_response is fired (PUSH 0x8D).
        """
        contact = self.contacts.get_by_key(pub_key)
        if not contact:
            return SentResult(success=False)
        # Resolve by exact public key, not name: two contacts can share a name
        # (e.g. a re-keyed node) and get_by_name returns the first match, which
        # would encrypt/route to the wrong key.
        proxy = self.contacts.get_proxy_by_key(pub_key)
        if not proxy:
            return SentResult(success=False)
        tag_int = random.randint(0, 0xFFFFFFFF)
        tag_bytes = tag_int.to_bytes(4, "little")
        inv_perm = 0xFF & ~TELEM_PERM_BASE
        req_payload = tag_bytes + bytes([REQ_TYPE_GET_TELEMETRY_DATA, inv_perm, 0, 0, 0])
        old_path_len = contact.out_path_len
        old_path = contact.out_path
        contact.out_path_len = -1
        contact.out_path = b""
        self.contacts.update(contact)
        try:
            pkt, _ = PacketBuilder.create_protocol_request(
                contact=proxy,
                local_identity=self._identity,
                protocol_code=REQ_TYPE_GET_TELEMETRY_DATA,
                data=req_payload,
            )
            self._apply_flood_scope(pkt)
            self._apply_path_hash_mode(pkt)
            success = await self._send_packet(pkt, wait_for_ack=False)
            if success:
                self._pending_discovery_tags.add(tag_int)
            return SentResult(
                success=success,
                is_flood=True,
                expected_ack=tag_int,
                timeout_ms=DEFAULT_RESPONSE_TIMEOUT_MS,
            )
        except Exception as e:
            logger.error("Error in path discovery: %s", e)
            return SentResult(success=False)
        finally:
            current = self.contacts.get_by_key(pub_key)
            if current and current.out_path_len == -1:
                current.out_path_len = old_path_len
                current.out_path = old_path
                self.contacts.update(current)

    async def send_text_message(
        self,
        pub_key: bytes,
        text: str,
        txt_type: int = TXT_TYPE_PLAIN,
        attempt: int = 1,
        wait_for_ack: bool = True,
        timestamp: Optional[int] = None,
    ) -> SentResult:
        """Send a direct text message to a contact.

        When wait_for_ack is True (default), blocks until ACK or timeout.
        When wait_for_ack is False, returns as soon as the packet is handed off;
        ACK (if any) is still tracked and will trigger send_confirmed later.
        For ``txt_type == TXT_TYPE_CLI_DATA``, delivery ACK is not used on MeshCore
        repeaters; ``wait_for_ack`` is treated as False and pending ACK is not tracked.
        """
        contact = self.contacts.get_by_key(pub_key)
        if not contact:
            logger.warning("Contact not found for key %s...", pub_key.hex()[:12])
            return SentResult(success=False)
        # Resolve by exact public key, not name: two contacts can share a name
        # (e.g. a re-keyed node) and get_by_name returns the first match, which
        # would encrypt/route to the wrong key.
        proxy = self.contacts.get_proxy_by_key(pub_key)
        if not proxy:
            return SentResult(success=False)
        try:
            is_flood = proxy.out_path_len < 0
            msg_type = "flood" if is_flood else "direct"
            pkt, ack_crc = PacketBuilder.create_text_message(
                contact=proxy,
                local_identity=self._identity,
                message=text,
                attempt=attempt,
                message_type=msg_type,
                txt_type=txt_type,
                timestamp=timestamp,
            )
            self._apply_flood_scope(pkt)
            self._apply_path_hash_mode(pkt)
            effective_wait_ack = wait_for_ack and txt_type != TXT_TYPE_CLI_DATA
            if txt_type != TXT_TYPE_CLI_DATA:
                self._track_pending_ack(ack_crc)
            if effective_wait_ack:
                success = await self._send_packet(pkt, wait_for_ack=True)
                if success:
                    self.stats.record_tx(is_flood=is_flood)
                else:
                    self.stats.record_tx_error()
                return SentResult(
                    success=success,
                    is_flood=is_flood,
                    expected_ack=ack_crc,
                    timeout_ms=None,
                )
            success = await self._send_packet(pkt, wait_for_ack=False)
            if success:
                self.stats.record_tx(is_flood=is_flood)
            else:
                self.stats.record_tx_error()
            return SentResult(
                success=success,
                is_flood=is_flood,
                expected_ack=ack_crc,
                timeout_ms=DEFAULT_RESPONSE_TIMEOUT_MS,
            )
        except Exception as e:
            logger.error("Error sending text message: %s", e)
            self.stats.record_tx_error()
            return SentResult(success=False)

    async def send_channel_message(self, channel_idx: int, text: str) -> bool:
        """Send a message to a channel."""
        channel = self.channels.get(channel_idx)
        if not channel:
            logger.warning("Channel %s not found", channel_idx)
            return False
        try:
            pkt = PacketBuilder.create_group_datagram(
                group_name=channel.name,
                local_identity=self._identity,
                message=text,
                sender_name=self.prefs.node_name,
                channels_config=self.channels.get_channels(),
            )
            self._apply_flood_scope(pkt)
            self._apply_path_hash_mode(pkt)
            success = await self._send_packet(pkt, wait_for_ack=False)
            if success:
                self.stats.record_tx(is_flood=True)
            else:
                self.stats.record_tx_error()
            return success
        except Exception as e:
            logger.error("Error sending channel message: %s", e)
            self.stats.record_tx_error()
            return False

    async def send_channel_data(
        self,
        channel_idx: int,
        data_type: int,
        payload: bytes,
        *,
        path: Optional[bytes] = None,
        path_len_encoded: Optional[int] = None,
    ) -> bool:
        """Send a group binary datagram (PAYLOAD_TYPE_GRP_DATA)."""
        channel = self.channels.get(channel_idx)
        if not channel or data_type <= 0 or data_type > 0xFFFF:
            return False
        payload = bytes(payload or b"")
        if len(payload) > 255:
            return False
        try:
            secret_bytes = bytes(channel.secret or b"")
            if len(secret_bytes) < 32:
                secret_bytes = secret_bytes + b"\x00" * (32 - len(secret_bytes))
            else:
                secret_bytes = secret_bytes[:32]

            hash_input = (
                secret_bytes[:16]
                if len(secret_bytes) >= 32 and secret_bytes[16:32] == b"\x00" * 16
                else secret_bytes
            )
            channel_hash = hashlib.sha256(hash_input).digest()[0]
            plaintext = struct.pack("<HB", data_type & 0xFFFF, len(payload)) + payload
            pkt = PacketBuilder.create_group_data_packet(
                PAYLOAD_TYPE_GRP_DATA,
                channel_hash,
                secret_bytes,
                plaintext,
                secret_bytes,
            )

            is_flood = path_len_encoded in (None, 0xFF)
            if is_flood:
                self._apply_flood_scope(pkt)
            else:
                pkt.header = (pkt.header & ~PH_ROUTE_MASK) | ROUTE_TYPE_DIRECT
                pkt.set_path(path or b"", path_len_encoded=path_len_encoded)
            self._apply_path_hash_mode(pkt)

            success = await self._send_packet(pkt, wait_for_ack=False)
            if success:
                self.stats.record_tx(is_flood=is_flood)
            else:
                self.stats.record_tx_error()
            return success
        except Exception as e:
            logger.error("Error sending channel data: %s", e)
            self.stats.record_tx_error()
            return False

    async def send_raw_data(
        self,
        dest_key: bytes,
        data: bytes,
        path: Optional[bytes] = None,
    ) -> SentResult:
        """Send raw data to a contact via a protocol request."""
        contact = self.contacts.get_by_key(dest_key)
        if not contact:
            return SentResult(success=False)
        # Resolve the proxy by the exact public key, not by name: two contacts can
        # share a name (e.g. a node that re-keyed) and get_by_name returns the first
        # match, which would encrypt/route to the wrong key.
        proxy = self.contacts.get_proxy_by_key(dest_key)
        if not proxy:
            return SentResult(success=False)
        try:
            pkt, _ = PacketBuilder.create_protocol_request(
                contact=proxy,
                local_identity=self._identity,
                protocol_code=PROTOCOL_CODE_RAW_DATA,
                data=data,
            )
            self._apply_path_hash_mode(pkt)
            success = await self._send_packet(pkt, wait_for_ack=False)
            return SentResult(success=success)
        except Exception as e:
            logger.error("Error sending raw data: %s", e)
            return SentResult(success=False)

    async def send_raw_data_direct(
        self, path: bytes, payload: bytes, *, path_len_encoded: int = None
    ) -> SentResult:
        """Send a raw custom packet (PAYLOAD_TYPE_RAW_CUSTOM) on the given direct path.

        No encryption or contact lookup; path and payload are supplied by the caller.
        Matches firmware CMD_SEND_RAW_DATA behaviour.

        Args:
            path_len_encoded: Encoded path_len byte. If None, assumes 1-byte hashes.
        """
        if len(payload) < 4:
            return SentResult(success=False)
        if len(path) > MAX_PATH_SIZE:
            return SentResult(success=False)
        if len(payload) > MAX_PACKET_PAYLOAD:
            return SentResult(success=False)
        try:
            pkt = PacketBuilder.create_raw_data(payload)
            pkt.set_path(path, path_len_encoded)
            success = await self._send_packet(pkt, wait_for_ack=False)
            if success:
                self.stats.record_tx(is_flood=False)
            else:
                self.stats.record_tx_error()
            return SentResult(success=success)
        except Exception as e:
            logger.error("Error sending raw data direct: %s", e)
            return SentResult(success=False)

    async def send_raw_packet(self, priority: int, packet_bytes: bytes) -> bool:
        """Inject a fully-formed on-air packet for transmission (CMD_SEND_RAW_PACKET).

        Mirrors firmware ``MyMesh.cpp`` ``CMD_SEND_RAW_PACKET``: parse the raw
        on-air bytes into a :class:`Packet` (``tryParsePacket``) and enqueue it
        for TX (``sendPacket``).  ``packet_bytes`` is the complete wire packet
        (header, optional transport codes, path, payload) as produced by
        :meth:`Packet.write_to`; it is sent verbatim, with no encryption,
        contact lookup, flood-scope, or path-hash-mode rewriting.

        The ``priority`` argument is accepted for protocol compatibility but is
        currently ignored: the bridge's low-level send path
        (:meth:`_send_packet`) does not expose a prioritized TX queue.

        Returns True if the packet parsed and was handed off for transmission,
        False on parse failure or send error (the frame_server handler maps
        False to ``ERR_CODE_TABLE_FULL``).
        """
        try:
            pkt = Packet()
            if not pkt.read_from(bytes(packet_bytes)):
                return False
        except Exception as e:
            logger.warning("send_raw_packet: failed to parse packet: %s", e)
            return False
        try:
            success = await self._send_packet(pkt, wait_for_ack=False)
            if success:
                self.stats.record_tx(is_flood=False)
            else:
                self.stats.record_tx_error()
            return success
        except Exception as e:
            logger.error("Error sending raw packet: %s", e)
            self.stats.record_tx_error()
            return False

    async def send_trace_path(
        self,
        pub_key: bytes,
        tag: int,
        auth_code: int,
        flags: int = 0,
    ) -> bool:
        """Send a trace path request to a contact."""
        contact = self.contacts.get_by_key(pub_key)
        if not contact:
            return False
        path = list(contact.out_path) if contact.out_path else []
        if not path:
            path = [contact.public_key[0]]
        try:
            pkt = PacketBuilder.create_trace(tag, auth_code, flags, path=path)
            self._apply_path_hash_mode(pkt)
            return await self._send_packet(pkt, wait_for_ack=False)
        except Exception as e:
            logger.error("Error sending trace: %s", e)
            return False

    async def send_control_data(self, data: Any = None) -> bool:
        """Send a CONTROL packet (e.g. discovery request).

        If *data* is provided it must be 1-254 bytes with the first byte having
        the 0x80 bit set (e.g. ``DISCOVER_REQ``).  Returns ``False`` for
        invalid payloads.

        When called with no *data* (or ``None``), a default discovery request
        is sent for backward compatibility.
        """
        try:
            if data and len(data) <= 254 and (data[0] & 0x80) != 0:
                pkt = Packet()
                pkt.header = PacketBuilder._create_header(PAYLOAD_TYPE_CONTROL, route_type="direct")
                pkt.path_len = 0
                pkt.path = bytearray()
                pkt.payload = bytearray(data)
                pkt.payload_len = len(data)
                self._apply_path_hash_mode(pkt)
                return await self._send_packet(pkt, wait_for_ack=False)
            elif data is not None:
                # data was provided but invalid
                return False
            # No data: send default discovery request
            tag = random.randint(0, 0xFFFFFFFF)
            pkt = PacketBuilder.create_discovery_request(tag, filter_mask=0x04)
            self._apply_path_hash_mode(pkt)
            return await self._send_packet(pkt, wait_for_ack=False)
        except Exception as e:
            logger.error("Error sending control data: %s", e)
            return False

    async def send_login(self, pub_key: bytes, password: str) -> dict:
        """Send a login request to a repeater and wait for the response."""
        contact = self.contacts.get_by_key(pub_key)
        if not contact:
            return {"success": False, "reason": "Contact not found"}
        # Resolve by exact public key, not name: two contacts can share a name
        # (e.g. a re-keyed node) and get_by_name returns the first match, which
        # would encrypt/route to the wrong key.
        proxy = self.contacts.get_proxy_by_key(pub_key)
        if not proxy:
            return {"success": False, "reason": "Contact not found"}
        login_handler = self._get_login_response_handler()
        if not login_handler:
            return {"success": False, "reason": "Login handler not available"}
        dest_hash = proxy.dest_hash
        login_handler.store_login_password(dest_hash, password)
        login_result: dict = {"success": False, "data": {}}
        login_event = asyncio.Event()

        def _login_cb(success: bool, data: dict) -> None:
            login_result["success"] = success
            login_result["data"] = data
            login_event.set()

        login_handler.set_login_callback(_login_cb)
        try:
            # The login callback fires on any decryptable login response from this
            # repeater (keyed by password/dest_hash, not by tag), so we can resend
            # a freshly-built login packet each attempt and a single event resolves
            # whichever attempt's reply arrives.
            async def _wait_login(timeout_s: float) -> dict:
                try:
                    await asyncio.wait_for(login_event.wait(), timeout=timeout_s)
                    return {"timeout": False}
                except asyncio.TimeoutError:
                    return {"timeout": True}

            await self._request_with_retries(
                lambda: PacketBuilder.create_login_packet(
                    contact=proxy, local_identity=self._identity, password=password
                ),
                _wait_login,
                proxy,
                log_label=f"login -> 0x{dest_hash:02X} ({contact.name})",
            )
            if not login_event.is_set():
                return {"success": False, "reason": "Login response timeout"}
            data = login_result["data"]
            return {
                "success": login_result["success"],
                "repeater": contact.name,
                "is_admin": data.get("is_admin", False),
                "keep_alive_interval": data.get("keep_alive_interval", 0),
                "tag": data.get("timestamp", 0),
                "acl_permissions": data.get("reserved", data.get("permissions", 0)),
                "firmware_ver_level": data.get("firmware_ver_level"),
                "reason": "Login successful" if login_result["success"] else "Login failed",
            }
        except Exception as e:
            logger.error("Login error: %s", e)
            return {"success": False, "reason": str(e)}
        finally:
            login_handler.set_login_callback(None)
            login_handler.clear_login_password(dest_hash)

    async def send_logout(self, pub_key: bytes) -> bool:
        """Send a logout / disconnect to a repeater contact."""
        contact = self.contacts.get_by_key(pub_key)
        if not contact:
            return False
        try:
            pkt, _ = PacketBuilder.create_logout_packet(
                contact=contact, local_identity=self._identity
            )
            self._apply_path_hash_mode(pkt)
            await self._send_packet(pkt, wait_for_ack=False)
            return True
        except Exception as e:
            logger.error("Logout error: %s", e)
            return False

    def _response_timeout_s(self, pkt: Packet, proxy: Any) -> float:
        """Adaptive response timeout (seconds) for a request packet.

        Mirrors firmware calcFloodTimeoutMillisFor / calcDirectTimeoutMillisFor
        using the radio's SF/BW/CR and the packet's on-air length, so a lost
        round-trip is retried on a ~3s cadence instead of a fixed 10-15s wait.
        """
        try:
            out_path_len = getattr(proxy, "out_path_len", -1)
            ms = response_timeout_ms(
                raw_length=pkt.get_raw_length(),
                is_flood=pkt.is_route_flood(),
                out_path_len=out_path_len,
                sf=int(getattr(self.prefs, "spreading_factor", 10)),
                bw_hz=int(getattr(self.prefs, "bandwidth_hz", 250000)),
                cr=int(getattr(self.prefs, "coding_rate", 5)),
            )
            return ms / 1000.0
        except Exception:
            return 5.0  # safe fallback

    async def _request_with_retries(
        self,
        build_packet: Callable[[], Packet],
        wait_for_response: Callable[[float], Awaitable[dict]],
        proxy: Any,
        *,
        total_timeout_s: Optional[float] = None,
        log_label: str = "request",
    ) -> dict:
        """Send a request up to DEFAULT_MAX_ATTEMPTS times until a response lands.

        A fresh packet is built per attempt (dodging repeater flood dedup) and
        each attempt waits one adaptive timeout (firmware cadence). A late reply
        that lands between attempts resolves the waiter immediately.

        ``total_timeout_s`` caps the cumulative wait across attempts: the final
        attempt's wait is clipped to the remaining budget and no new attempt
        starts once the budget is spent.
        """
        result: dict = {"timeout": True}
        deadline = time.monotonic() + total_timeout_s if total_timeout_s else None
        for attempt in range(DEFAULT_MAX_ATTEMPTS):
            pkt = build_packet()
            self._apply_path_hash_mode(pkt)
            timeout_s = self._response_timeout_s(pkt, proxy)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                timeout_s = min(timeout_s, remaining)
            logger.debug(
                "[PATHDIAG] %s: route=%s attempt=%d/%d timeout=%.1fs out_path_len=%s",
                log_label,
                "FLOOD" if pkt.is_route_flood() else "DIRECT",
                attempt + 1,
                DEFAULT_MAX_ATTEMPTS,
                timeout_s,
                getattr(proxy, "out_path_len", -1),
            )
            await self._send_packet(pkt, wait_for_ack=False)
            result = await wait_for_response(timeout_s)
            if not result.get("timeout"):
                break
        return result

    async def _wait_for_path_propagation(self, proxy: Any, request_type: str) -> None:
        """Log the pre-send path; no longer sleeps.

        Firmware sends the request immediately and relies on the reciprocal PATH
        (which openHop already sends at login time, see ProtocolResponseHandler).
        The previous 0.5s/hop sleep added up to ~1.5s+ of latency per request for
        multi-hop contacts with no reliability benefit and has been removed; the
        adaptive timeout + internal resend now handle a lost first attempt.
        """
        out_path_len = getattr(proxy, "out_path_len", -1)
        out_path = getattr(proxy, "out_path", b"") or b""
        logger.debug(
            "[PATHDIAG] %s pre-send: %s",
            request_type,
            _fmt_path(out_path_len, out_path),
        )

    async def send_status_request(self, pub_key: bytes, timeout: float = 15.0) -> dict:
        """Send a protocol request for repeater status/stats.

        ``timeout`` caps the total wait across retries (seconds); each attempt
        still uses the adaptive per-attempt timeout.
        """
        contact = self.contacts.get_by_key(pub_key)
        if not contact:
            return {"success": False, "reason": "Contact not found"}
        # Resolve by exact public key, not name: two contacts can share a name
        # (e.g. a re-keyed node) and get_by_name returns the first match, which
        # would encrypt/route to the wrong key.
        proxy = self.contacts.get_proxy_by_key(pub_key)
        if not proxy:
            return {"success": False, "reason": "Contact not found"}
        proto_handler = self._get_protocol_response_handler()
        if not proto_handler:
            return {"success": False, "reason": "Protocol handler not available"}
        contact_hash = proxy.dest_hash
        waiter = ResponseWaiter()
        proto_handler.set_response_callback(contact_hash, waiter.callback)
        try:
            await self._wait_for_path_propagation(proxy, "stats request")
            # Status responses resolve the waiter by contact_hash (not tag), so a
            # fresh REQ each attempt is fine and dodges the repeater's flood dedup.
            result = await self._request_with_retries(
                lambda: PacketBuilder.create_protocol_request(
                    contact=proxy,
                    local_identity=self._identity,
                    protocol_code=REQ_TYPE_GET_STATUS,
                    data=b"",
                )[0],
                waiter.wait,
                proxy,
                total_timeout_s=timeout,
                log_label="stats REQ",
            )
            return {
                "success": result.get("success", False),
                "repeater": contact.name,
                "stats": result.get("parsed", {}),
                "response_text": result.get("text"),
                "reason": "Stats received" if result.get("success") else "Stats request failed",
            }
        except Exception as e:
            logger.error("Status request error: %s", e)
            return {"success": False, "reason": str(e)}
        finally:
            proto_handler.clear_response_callback(contact_hash)

    async def send_telemetry_request(
        self,
        pub_key: bytes,
        want_base: bool = True,
        want_location: bool = True,
        want_environment: bool = True,
        timeout: float = 10.0,
    ) -> dict:
        """Send a telemetry request to a contact and wait for the response.

        ``timeout`` caps the total wait across retries (seconds); each attempt
        still uses the adaptive per-attempt timeout.
        """
        contact = self.contacts.get_by_key(pub_key)
        if not contact:
            return {"success": False, "reason": "Contact not found"}
        # Resolve by exact public key, not name: two contacts can share a name
        # (e.g. a re-keyed node) and get_by_name returns the first match, which
        # would encrypt/route to the wrong key.
        proxy = self.contacts.get_proxy_by_key(pub_key)
        if not proxy:
            return {"success": False, "reason": "Contact not found"}
        proto_handler = self._get_protocol_response_handler()
        if not proto_handler:
            return {"success": False, "reason": "Protocol handler not available"}
        contact_hash = proxy.dest_hash
        waiter = ResponseWaiter()
        proto_handler.set_response_callback(contact_hash, waiter.callback)
        try:
            await self._wait_for_path_propagation(proxy, "telemetry request")
            inv = PacketBuilder._compute_inverse_perm_mask(
                want_base, want_location, want_environment
            )
            result = await self._request_with_retries(
                lambda: PacketBuilder.create_protocol_request(
                    contact=proxy,
                    local_identity=self._identity,
                    protocol_code=REQ_TYPE_GET_TELEMETRY_DATA,
                    data=bytes([inv]),
                )[0],
                waiter.wait,
                proxy,
                total_timeout_s=timeout,
                log_label="telemetry REQ",
            )
            telemetry_data = dict(result.get("parsed", {}))
            raw_bytes = telemetry_data.get("raw_bytes", b"")
            if raw_bytes and len(pub_key) >= 6:
                # Companion-style frame: 0x8B + reserved + 6-byte pubkey prefix + LPP
                telemetry_data["frame_bytes"] = (
                    bytes([PUSH_CODE_TELEMETRY_RESPONSE, 0]) + pub_key[:6] + raw_bytes
                )
            return {
                "success": result.get("success", False),
                "contact": contact.name,
                "telemetry_data": telemetry_data,
                "response_text": result.get("text"),
                "reason": ("Telemetry received" if result.get("success") else "Telemetry failed"),
            }
        except Exception as e:
            logger.error("Telemetry error: %s", e)
            return {"success": False, "reason": str(e)}
        finally:
            proto_handler.clear_response_callback(contact_hash)

    async def send_binary_request(self, pub_key: bytes, data: bytes) -> dict:
        """Legacy: send binary request and wait.

        Prefer ``send_binary_req`` + ``on_binary_response``.
        """
        return await self._send_protocol_request(pub_key, PROTOCOL_CODE_BINARY_REQ, data)

    async def send_anon_request(self, pub_key: bytes, data: bytes) -> dict:
        """Send an anonymous request to a contact and wait for the response."""
        return await self._send_protocol_request(pub_key, PROTOCOL_CODE_ANON_REQ, data)

    async def _send_protocol_request(self, pub_key: bytes, protocol_code: int, data: bytes) -> dict:
        """Build and send a protocol request, waiting for the response."""
        contact = self.contacts.get_by_key(pub_key)
        if not contact:
            return {"success": False, "reason": "Contact not found"}
        # Resolve by exact public key, not name: two contacts can share a name
        # (e.g. a re-keyed node) and get_by_name returns the first match, which
        # would encrypt/route to the wrong key.
        proxy = self.contacts.get_proxy_by_key(pub_key)
        if not proxy:
            return {"success": False, "reason": "Contact not found"}
        proto_handler = self._get_protocol_response_handler()
        if not proto_handler:
            return {"success": False, "reason": "Protocol handler not available"}
        contact_hash = proxy.dest_hash
        waiter = ResponseWaiter()
        proto_handler.set_response_callback(contact_hash, waiter.callback)
        try:
            result = await self._request_with_retries(
                lambda: PacketBuilder.create_protocol_request(
                    contact=proxy,
                    local_identity=self._identity,
                    protocol_code=protocol_code,
                    data=data,
                )[0],
                waiter.wait,
                proxy,
                log_label=f"protocol REQ 0x{protocol_code:02X}",
            )
            return {
                "success": result.get("success", False),
                "response": result.get("text"),
                "parsed_data": result.get("parsed", {}),
                "reason": "Success" if result.get("success") else "Failed",
            }
        except Exception as e:
            logger.error("Protocol request error: %s", e)
            return {"success": False, "reason": str(e)}
        finally:
            proto_handler.clear_response_callback(contact_hash)

    async def send_repeater_command(
        self, pub_key: bytes, command: str, parameters: Optional[str] = None
    ) -> dict:
        """Send a text-based command to a repeater and wait for the response."""
        contact = self.contacts.get_by_key(pub_key)
        if not contact:
            return {"success": False, "reason": "Contact not found"}
        # Resolve by exact public key, not name: two contacts can share a name
        # (e.g. a re-keyed node) and get_by_name returns the first match, which
        # would encrypt/route to the wrong key.
        proxy = self.contacts.get_proxy_by_key(pub_key)
        if not proxy:
            return {"success": False, "reason": "Contact not found"}
        text_handler = self._get_text_handler()
        if not text_handler:
            return {"success": False, "reason": "Text handler not available"}
        full_command = command
        if parameters:
            full_command += f" {parameters}"
        response_data: dict = {"text": None, "success": False}
        response_event = asyncio.Event()

        def _response_cb(message_text: str, sender_contact: Any) -> None:
            response_data["text"] = message_text
            response_data["success"] = True
            response_event.set()

        text_handler.set_command_response_callback(_response_cb)
        try:
            msg_type = "flood" if proxy.out_path_len < 0 else "direct"
            pkt, _ = PacketBuilder.create_text_message(
                contact=proxy,
                local_identity=self._identity,
                message=full_command,
                attempt=1,
                message_type=msg_type,
                txt_type=TXT_TYPE_CLI_DATA,
            )
            self._apply_path_hash_mode(pkt)
            await self._send_packet(pkt, wait_for_ack=False)
            try:
                await asyncio.wait_for(response_event.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                pass
            return {
                "success": response_data["success"],
                "repeater": contact.name,
                "command": command,
                "response": response_data["text"],
                "reason": ("Command successful" if response_data["success"] else "No response"),
            }
        except Exception as e:
            logger.error("Repeater command error: %s", e)
            return {"success": False, "reason": str(e)}
        finally:
            text_handler.set_command_response_callback(None)

    def _track_pending_ack(self, ack_crc: int) -> None:
        """Track pending ACK CRC for send_confirmed (capped)."""
        if len(self._pending_ack_crcs) < MAX_PENDING_ACK_CRCS:
            self._pending_ack_crcs.add(ack_crc)

    async def _try_confirm_send(self, crc: int) -> bool:
        """If CRC is pending, discard it and fire send_confirmed. Returns True if fired."""
        if crc not in self._pending_ack_crcs:
            return False
        self._pending_ack_crcs.discard(crc)
        await self._fire_callbacks("send_confirmed", crc)
        return True

    def sync_next_message(self) -> Optional[QueuedMessage]:
        """Pop and return the next queued message, or None."""
        return self.message_queue.pop()
