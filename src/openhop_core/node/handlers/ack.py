from __future__ import annotations

import inspect
from typing import Awaitable, Callable, Optional

from ...protocol import Packet
from ...protocol.constants import PAYLOAD_TYPE_ACK
from ...protocol.packet_utils import PathUtils
from .base import BaseHandler
from .crypto_helpers import iter_decrypt_by_src_hash


class AckHandler(BaseHandler):
    """
    ACK handler that processes all ACK variants:
    1. Discrete ACK packets (payload type 1)
    2. Bundled ACKs in PATH packets
    3. Encrypted ACK responses carried by PATH packets
    """

    @staticmethod
    def payload_type() -> int:
        return PAYLOAD_TYPE_ACK

    def __init__(self, log_fn, dispatcher=None):
        self.log = log_fn
        self.dispatcher = dispatcher
        self._ack_received_callback: Optional[Callable[[int], Awaitable[None] | None]] = None

    def set_ack_received_callback(
        self, callback: Optional[Callable[[int], Awaitable[None] | None]]
    ):
        """Set callback to notify dispatcher when ACK is received."""
        self._ack_received_callback = callback

    def set_dispatcher(self, dispatcher):
        """Set dispatcher reference for contact lookup and waiting ACKs."""
        self.dispatcher = dispatcher

    async def __call__(self, packet: Packet) -> None:
        """Handle discrete ACK packets (payload type 1)."""
        ack_crc = await self.process_discrete_ack(packet)
        if ack_crc is not None:
            await self._notify_ack_received(ack_crc)

    async def process_discrete_ack(self, packet: Packet) -> Optional[int]:
        """Process a discrete ACK packet and return the CRC if valid."""
        self.log(f"Processing discrete ACK: payload_len={len(packet.payload)}")
        self.log(f"ACK payload (hex): {packet.payload.hex().upper()}")

        if len(packet.payload) < 4:
            self.log(f"Invalid ACK length: {len(packet.payload)} bytes (expected >= 4)")
            return None

        # Extract CRC checksum from the first 4 bytes (little endian per protocol spec).
        # Firmware emits 6-byte ACKs for plain DMs (4-byte hash + ext-attempt + random byte);
        # only the first 4 bytes are matched against the expected ACK.
        crc = int.from_bytes(packet.payload[:4], "little")
        self.log(f"Discrete ACK received: CRC={crc:08X}")
        return crc

    async def process_path_ack_variants(self, packet: Packet) -> Optional[int]:
        """
        Process PATH packets that may contain ACKs in different forms.
        Returns CRC if ACK found, None otherwise.
        """
        if not self.dispatcher:
            return None

        payload = packet.payload
        if len(payload) < 1:
            return None

        self.log(f"Processing PATH packet for ACKs: payload_len={len(payload)}")
        self.log(f"PATH payload (hex): {payload.hex().upper()}")

        # PATH returns are encrypted as dest_hash + src_hash + MAC + ciphertext.
        # Their outer length varies with the returned path, so do not restrict this
        # to the 20-byte (single AES-block) form.
        if (
            self.dispatcher._waiting_acks
            and self.dispatcher.local_identity
            and self.dispatcher.contact_book
            and len(payload) >= 2
            and payload[0] == self.dispatcher.local_identity.get_public_key()[0]
        ):
            self.log("Checking encrypted PATH packet for ACK response")
            ack_crc = await self._try_decrypt_encrypted_ack(payload)
            if ack_crc is not None:
                self.log(f"Found encrypted ACK response: CRC={ack_crc:08X}")
                return ack_crc

        return None

    async def _try_decrypt_encrypted_ack(self, payload: bytes) -> Optional[int]:
        """Decrypt an addressed PATH return and extract its ACK extra, if any.

        A PATH source hash is only one byte, so it identifies a candidate set rather
        than a unique contact.  A valid MAC identifies the actual sender.  After a
        successful decrypt, decode the inner PATH layout instead of searching its
        path or non-ACK extra bytes for a value that happens to match a pending CRC.
        """
        if len(payload) < 2:
            return None

        src_hash = payload[1]
        encrypted = bytes(payload[2:])
        contacts = getattr(self.dispatcher.contact_book, "contacts", ())

        for _contact, _pubkey, _secret, decrypted in iter_decrypt_by_src_hash(
            contacts, src_hash, self.dispatcher.local_identity, encrypted
        ):
            # MeshCore treats a successfully authenticated PATH as belonging to
            # that matched contact.  Reject a malformed inner PATH rather than
            # interpreting arbitrary bytes as an ACK.
            if not decrypted or not PathUtils.is_valid_path_len(decrypted[0]):
                self.log("Encrypted PATH ACK has an invalid path length")
                return None

            path_byte_len = PathUtils.get_path_byte_len(decrypted[0])
            extra_start = 1 + path_byte_len
            if len(decrypted) < extra_start + 1:
                self.log("Encrypted PATH ACK is truncated before its extra type")
                return None

            extra_type = decrypted[extra_start] & 0x0F
            if extra_type != PAYLOAD_TYPE_ACK:
                return None

            if len(decrypted) < extra_start + 5:
                self.log("Encrypted PATH ACK extra is shorter than its CRC")
                return None

            crc = int.from_bytes(decrypted[extra_start + 1 : extra_start + 5], "little")
            if crc in self.dispatcher._waiting_acks:
                return crc
            return None

        return None

    async def _notify_ack_received(self, crc: int):
        """Notify the dispatcher that an ACK was received."""
        if self._ack_received_callback:
            cb = self._ack_received_callback
            if inspect.iscoroutinefunction(cb):
                await cb(crc)
            else:
                cb(crc)
