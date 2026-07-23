"""Return-path teaching parity with MeshCore's BaseChatMesh::handleReturnPathRetry.

A MeshCore server answers a DIRECT request from the ``out_path`` stored in its
ACL, never by reversing the inbound path. A client that never teaches that route
leaves the server replying into a dead route, which is what breaks CLI/protocol
requests over a user-forced path: the login goes out DIRECT, so the server
answers with a plain flood RESPONSE rather than the flood PATH that normally
carries the reciprocal, and nothing else in the stack teaches the route back.
"""

import asyncio

import pytest

from openhop_core.companion.base_send import _SendOpsMixin
from openhop_core.companion.contact_store import ContactStore
from openhop_core.companion.models import Contact
from openhop_core.node.handlers.login_response import LoginResponseHandler
from openhop_core.node.handlers.protocol_response import ProtocolResponseHandler
from openhop_core.node.handlers.registry import create_core_handlers
from openhop_core.node.handlers.return_path import ReturnPathTeacher, reverse_path
from openhop_core.protocol import CryptoUtils, Identity, LocalIdentity, Packet
from openhop_core.protocol.constants import (
    PAYLOAD_TYPE_PATH,
    PAYLOAD_TYPE_RESPONSE,
    ROUTE_TYPE_DIRECT,
    ROUTE_TYPE_FLOOD,
)
from openhop_core.protocol.packet_utils import PathUtils

LOCAL_IDENTITY = LocalIdentity(bytes(32))
PEER = LocalIdentity(bytes([0x5A]) + bytes(31))
PEER_HASH = PEER.get_public_key()[0]
PEER_KEY = PEER.get_public_key()

# Route from us to the peer (what a user's "force path" sets).
OUT_PATH = bytes([0xAA, 0xBB])
OUT_PATH_LEN = PathUtils.encode_path_len(1, 2)

# Route the peer's flood reply accumulated on its way back to us. Deliberately
# different from OUT_PATH so a test can tell the embedded path (peer -> us) from
# the routing path (us -> peer); swapping the two is the classic failure here.
IN_PATH = bytes([0xCC, 0xDD])
IN_PATH_LEN = PathUtils.encode_path_len(1, 2)


def _shared_secret():
    return Identity(PEER_KEY).calc_shared_secret(LOCAL_IDENTITY.get_private_key())


def _contacts(out_path=OUT_PATH, out_path_len=OUT_PATH_LEN):
    contacts = ContactStore()
    contact = Contact(public_key=PEER_KEY, name="FarRepeater")
    contact.out_path = out_path
    contact.out_path_len = out_path_len
    contacts.add(contact)
    return contacts


def _response_packet(*, flood: bool, in_path=IN_PATH, in_path_len=IN_PATH_LEN, tag=0x11223344):
    """A PAYLOAD_TYPE_RESPONSE from PEER addressed to us."""
    secret = _shared_secret()
    plaintext = tag.to_bytes(4, "little") + b"ok"
    packet = Packet()
    packet.header = (PAYLOAD_TYPE_RESPONSE << 2) | (
        ROUTE_TYPE_FLOOD if flood else ROUTE_TYPE_DIRECT
    )
    packet.path = bytearray(in_path) if flood else bytearray()
    packet.path_len = in_path_len if flood else 0
    packet.payload = bytearray(
        bytes([LOCAL_IDENTITY.get_public_key()[0], PEER_HASH])
        + CryptoUtils.encrypt_then_mac(secret[:16], secret, plaintext)
    )
    packet.payload_len = len(packet.payload)
    return packet


def _decode_teach(packet: Packet):
    """Return (embedded_path_len_byte, embedded_path) from a teach packet."""
    secret = _shared_secret()
    plaintext = CryptoUtils.mac_then_decrypt(secret[:16], secret, bytes(packet.payload[2:]))
    assert plaintext, "teach packet did not authenticate against the peer secret"
    path_len_byte = plaintext[0]
    byte_len = PathUtils.get_path_byte_len(path_len_byte)
    return path_len_byte, bytes(plaintext[1 : 1 + byte_len])


class _Injector:
    """Captures injected packets; can be made to fail or to block.

    ``slow=True`` models a real transmit path (TX lock, airtime budget, on-air
    time) so tests can observe behaviour across the injector's await point.
    """

    def __init__(self, fail=False, slow=False):
        self.packets = []
        self.fail = fail
        self._gate = asyncio.Event() if slow else None

    def release(self) -> None:
        if self._gate is not None:
            self._gate.set()

    async def __call__(self, packet, *args, **kwargs):
        if self._gate is not None:
            await asyncio.sleep(0)  # let concurrent callers all reach here
            if not self._gate.is_set():
                await asyncio.wait_for(self._gate.wait(), timeout=2.0)
        if self.fail:
            raise RuntimeError("radio down")
        self.packets.append(packet)
        return True


def _teacher(contacts=None, injector=None, **kwargs):
    teacher = ReturnPathTeacher(
        lambda _m: None, LOCAL_IDENTITY, contacts if contacts is not None else _contacts(), **kwargs
    )
    teacher.set_injector(injector if injector is not None else _Injector())
    return teacher


async def _teach_flood(teacher, **kwargs):
    """Trigger a flood-reply teach and wait for it to reach the injector."""
    sent = await teacher.maybe_teach_from_flood_reply(
        _response_packet(flood=True, **kwargs), PEER_KEY, PEER_HASH, reason="test"
    )
    await teacher.wait_for_pending()
    return sent


# --------------------------------------------------------------------------- #
# reverse_path
# --------------------------------------------------------------------------- #


def test_reverse_path_reverses_whole_hashes_not_bytes():
    """A 2-byte-hash path must reverse hop order, keeping each hash intact."""
    path = bytes([0x11, 0x22, 0x33, 0x44])
    assert reverse_path(path, PathUtils.encode_path_len(2, 2)) == bytes([0x33, 0x44, 0x11, 0x22])
    # Byte-wise reversal would give 44332211 and corrupt every hop.


def test_reverse_path_three_byte_hashes():
    path = bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06])
    assert reverse_path(path, PathUtils.encode_path_len(3, 2)) == bytes(
        [0x04, 0x05, 0x06, 0x01, 0x02, 0x03]
    )


def test_reverse_path_single_byte_hashes():
    assert reverse_path(bytes([0xA1, 0xB2, 0xC3]), PathUtils.encode_path_len(1, 3)) == bytes(
        [0xC3, 0xB2, 0xA1]
    )


def test_reverse_path_zero_hop_is_empty_not_none():
    """A zero-hop path is valid (direct neighbour), distinct from malformed."""
    assert reverse_path(b"", 0) == b""


def test_reverse_path_rejects_length_mismatch():
    assert reverse_path(bytes([0xAA]), PathUtils.encode_path_len(1, 3)) is None


# --------------------------------------------------------------------------- #
# Flood-reply trigger (firmware parity)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_flood_reply_teaches_inbound_path_routed_via_out_path():
    """The teach embeds the peer->us path and is routed along the us->peer path."""
    injector = _Injector()
    teacher = _teacher(injector=injector)

    assert await _teach_flood(teacher) is True
    assert len(injector.packets) == 1
    teach = injector.packets[0]

    # Shape: a PATH packet sent DIRECT (firmware sendDirect of createPathReturn).
    assert teach.get_payload_type() == PAYLOAD_TYPE_PATH
    assert teach.get_route_type() == ROUTE_TYPE_DIRECT
    # Routed along OUR out_path so it actually reaches the peer.
    assert bytes(teach.path) == OUT_PATH
    assert teach.path_len == OUT_PATH_LEN
    # Addressed to the peer, from us.
    assert teach.payload[0] == PEER_HASH
    assert teach.payload[1] == LOCAL_IDENTITY.get_public_key()[0]
    # Embedded: the route the peer should use to reach us == the inbound path.
    embedded_len, embedded_path = _decode_teach(teach)
    assert embedded_path == IN_PATH
    assert embedded_len == IN_PATH_LEN


@pytest.mark.asyncio
async def test_flood_reply_teach_preserves_two_byte_hash_encoding():
    """path_len encodes hash_size in bits 6-7; a 2-byte-hash route must survive
    both the embedded payload and the outer routing path intact."""
    out_path = bytes([0x11, 0x22, 0x33, 0x44])
    out_len = PathUtils.encode_path_len(2, 2)
    in_path = bytes([0xA1, 0xA2, 0xB1, 0xB2])
    in_len = PathUtils.encode_path_len(2, 2)

    injector = _Injector()
    teacher = _teacher(contacts=_contacts(out_path, out_len), injector=injector)

    assert await _teach_flood(teacher, in_path=in_path, in_path_len=in_len) is True
    teach = injector.packets[0]
    assert bytes(teach.path) == out_path
    assert teach.path_len == out_len
    embedded_len, embedded_path = _decode_teach(teach)
    assert embedded_path == in_path
    assert embedded_len == in_len
    assert PathUtils.get_path_hash_size(embedded_len) == 2


@pytest.mark.asyncio
async def test_direct_reply_does_not_teach():
    """Only a flood reply signals the peer has no route back (firmware guard)."""
    injector = _Injector()
    teacher = _teacher(injector=injector)

    sent = await teacher.maybe_teach_from_flood_reply(
        _response_packet(flood=False), PEER_KEY, PEER_HASH, reason="test"
    )
    await teacher.wait_for_pending()

    assert sent is False
    assert injector.packets == []


@pytest.mark.asyncio
async def test_unknown_out_path_does_not_teach():
    """OUT_PATH_UNKNOWN (-1): flood replies are expected, and we have no route
    to send a direct teach down anyway."""
    injector = _Injector()
    teacher = _teacher(contacts=_contacts(out_path=b"", out_path_len=-1), injector=injector)

    assert await _teach_flood(teacher) is False
    assert injector.packets == []


@pytest.mark.asyncio
async def test_zero_hop_out_path_is_known_and_teaches():
    """out_path_len == 0 is a known zero-hop route, NOT unknown. Truth-testing
    the value instead of range-checking it would silently skip direct
    neighbours."""
    injector = _Injector()
    teacher = _teacher(contacts=_contacts(out_path=b"", out_path_len=0), injector=injector)

    assert await _teach_flood(teacher) is True
    assert bytes(injector.packets[0].path) == b""


@pytest.mark.asyncio
async def test_out_path_shorter_than_declared_length_is_rejected():
    """A truncated stored path must not be transmitted as a routing path."""
    injector = _Injector()
    teacher = _teacher(
        contacts=_contacts(out_path=bytes([0xAA]), out_path_len=PathUtils.encode_path_len(1, 3)),
        injector=injector,
    )
    assert await _teach_flood(teacher) is False
    assert injector.packets == []


@pytest.mark.asyncio
async def test_no_injector_is_a_no_op():
    teacher = ReturnPathTeacher(lambda _m: None, LOCAL_IDENTITY, _contacts())
    assert teacher.enabled is False
    assert await _teach_flood(teacher) is False


# --------------------------------------------------------------------------- #
# Rate limiting and dispatch
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cooldown_suppresses_immediate_second_teach():
    clock = {"t": 1000.0}
    injector = _Injector()
    teacher = _teacher(injector=injector, cooldown_s=5.0, time_fn=lambda: clock["t"])

    assert await _teach_flood(teacher) is True
    clock["t"] += 1.0
    assert await _teach_flood(teacher) is False
    clock["t"] += 5.0
    assert await _teach_flood(teacher) is True
    assert len(injector.packets) == 2


@pytest.mark.asyncio
async def test_concurrent_triggers_transmit_only_one_teach():
    """The cooldown is claimed synchronously, before anything is awaited, so two
    triggers racing across the injector's await point cannot both transmit."""
    injector = _Injector(slow=True)
    teacher = _teacher(injector=injector, cooldown_s=5.0)

    results = await asyncio.gather(
        *[
            teacher.maybe_teach_from_flood_reply(
                _response_packet(flood=True), PEER_KEY, PEER_HASH, reason="t"
            )
            for _ in range(3)
        ]
    )
    injector.release()
    await teacher.wait_for_pending()

    assert sum(1 for r in results if r) == 1
    assert len(injector.packets) == 1


@pytest.mark.asyncio
async def test_teach_does_not_block_on_the_injector():
    """Firmware queues this send; awaiting it inline would stall the RX path and
    delay the very reply that triggered the teach."""
    injector = _Injector(slow=True)
    teacher = _teacher(injector=injector)

    assert await teacher.maybe_teach_from_flood_reply(
        _response_packet(flood=True), PEER_KEY, PEER_HASH, reason="t"
    )
    # Returned while the injector is still blocked.
    assert injector.packets == []

    injector.release()
    await teacher.wait_for_pending()
    assert len(injector.packets) == 1


@pytest.mark.asyncio
async def test_failed_inject_releases_cooldown_for_the_next_trigger():
    """A teach that never made it onto the radio must be retried on the next
    trigger, not muted for the cooldown window."""
    clock = {"t": 0.0}
    teacher = _teacher(injector=_Injector(fail=True), cooldown_s=5.0, time_fn=lambda: clock["t"])

    await _teach_flood(teacher)

    working = _Injector()
    teacher.set_injector(working)
    assert await _teach_flood(teacher) is True
    assert len(working.packets) == 1


# --------------------------------------------------------------------------- #
# Reverse-of-out_path hardening (no firmware equivalent)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_reverse_teach_embeds_reversed_out_path():
    """With nothing inbound to learn from, assume a symmetric route. Uses 2-byte
    hashes so a byte-wise reversal would not pass."""
    out_path = bytes([0x11, 0x22, 0x33, 0x44])
    out_len = PathUtils.encode_path_len(2, 2)
    injector = _Injector()
    teacher = _teacher(contacts=_contacts(out_path, out_len), injector=injector)

    assert await teacher.maybe_teach_reverse_of_out_path(PEER_KEY, reason="timeout") is True
    await teacher.wait_for_pending()

    teach = injector.packets[0]
    assert teach.get_route_type() == ROUTE_TYPE_DIRECT
    assert bytes(teach.path) == out_path  # still routed outward along out_path
    _, embedded_path = _decode_teach(teach)
    assert embedded_path == bytes([0x33, 0x44, 0x11, 0x22])


@pytest.mark.asyncio
async def test_reverse_teach_skipped_without_known_out_path():
    injector = _Injector()
    teacher = _teacher(contacts=_contacts(out_path=b"", out_path_len=-1), injector=injector)
    assert not await teacher.maybe_teach_reverse_of_out_path(PEER_KEY, reason="timeout")
    assert injector.packets == []


@pytest.mark.asyncio
async def test_reverse_teach_never_overwrites_an_evidence_derived_teach():
    """THE regression guard. The peer applies whichever path it received last
    (MyMesh::onPeerPathRecv overwrites unconditionally). If a merely slow first
    attempt let the symmetry guess replace the route we learned from a real
    inbound path, and the guess is wrong, the peer would reply into a void — no
    flood reply would ever arrive again, so nothing could re-teach it. That is
    strictly worse than never having guessed."""
    clock = {"t": 0.0}
    injector = _Injector()
    teacher = _teacher(injector=injector, cooldown_s=5.0, time_fn=lambda: clock["t"])

    assert await _teach_flood(teacher) is True
    _, embedded = _decode_teach(injector.packets[0])
    assert embedded == IN_PATH

    # Well past the cooldown: only the evidence guard can stop this.
    clock["t"] += 3600.0
    assert await teacher.maybe_teach_reverse_of_out_path(PEER_KEY, reason="timeout") is False
    await teacher.wait_for_pending()
    assert len(injector.packets) == 1, "the symmetry guess clobbered a known-good route"


@pytest.mark.asyncio
async def test_note_evidence_teach_blocks_the_reverse_guess():
    injector = _Injector()
    teacher = _teacher(injector=injector)

    teacher.note_evidence_teach(PEER_KEY)

    assert await teacher.maybe_teach_reverse_of_out_path(PEER_KEY, reason="timeout") is False
    assert injector.packets == []


def _flood_path_packet(inner_path=OUT_PATH, inner_len=OUT_PATH_LEN):
    """A flood PAYLOAD_TYPE_PATH carrying a RESPONSE — what a *flood* login gets.

    Inner layout: path_len(1) + path(N) + extra_type(1) + extra.
    """
    secret = _shared_secret()
    response = (0x11223344).to_bytes(4, "little") + b"ok"
    inner = bytes([inner_len]) + inner_path + bytes([PAYLOAD_TYPE_RESPONSE]) + response
    packet = Packet()
    packet.header = (PAYLOAD_TYPE_PATH << 2) | ROUTE_TYPE_FLOOD
    packet.path = bytearray(IN_PATH)
    packet.path_len = IN_PATH_LEN
    packet.payload = bytearray(
        bytes([LOCAL_IDENTITY.get_public_key()[0], PEER_HASH])
        + CryptoUtils.encrypt_then_mac(secret[:16], secret, inner)
    )
    packet.payload_len = len(packet.payload)
    return packet


@pytest.mark.asyncio
async def test_flood_path_reciprocal_reports_itself_and_blocks_the_reverse_guess():
    """End-to-end: the pre-existing flood-PATH reciprocal teaches from a real
    inbound path. If ProtocolResponseHandler did not report it, a normal flood
    login would leave the symmetry guess free to overwrite a correct route on
    the very first retry — with no cooldown involved."""
    injector = _Injector()
    handler = ProtocolResponseHandler(lambda _m: None, LOCAL_IDENTITY, _contacts())
    handler.set_packet_injector(injector)

    await handler(_flood_path_packet())
    await handler.return_path_teacher.wait_for_pending()

    # Exactly one packet: the reciprocal. The RESPONSE-branch teach must not
    # also fire for a PATH packet.
    assert len(injector.packets) == 1

    blocked = await handler.return_path_teacher.maybe_teach_reverse_of_out_path(
        PEER_KEY, reason="timeout"
    )
    await handler.return_path_teacher.wait_for_pending()
    assert blocked is False
    assert len(injector.packets) == 1


# --------------------------------------------------------------------------- #
# End-to-end through the handlers
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_protocol_response_handler_teaches_on_flood_response():
    injector = _Injector()
    handler = ProtocolResponseHandler(lambda _m: None, LOCAL_IDENTITY, _contacts())
    handler.set_packet_injector(injector)

    await handler(_response_packet(flood=True))
    await handler.return_path_teacher.wait_for_pending()

    assert len(injector.packets) == 1
    assert injector.packets[0].get_payload_type() == PAYLOAD_TYPE_PATH
    _, embedded_path = _decode_teach(injector.packets[0])
    assert embedded_path == IN_PATH


@pytest.mark.asyncio
async def test_protocol_response_handler_does_not_teach_on_direct_response():
    injector = _Injector()
    handler = ProtocolResponseHandler(lambda _m: None, LOCAL_IDENTITY, _contacts())
    handler.set_packet_injector(injector)

    await handler(_response_packet(flood=False))
    await handler.return_path_teacher.wait_for_pending()

    assert injector.packets == []


def _login_reply_packet(flood=True):
    """Firmware handleLoginReq reply_data: timestamp(4) + RESP_SERVER_LOGIN_OK +
    keep_alive + is_admin + permissions + random(4) + firmware_ver_level."""
    login_reply = (
        (0x01020304).to_bytes(4, "little") + bytes([0x80, 0x00, 0x01, 0x03]) + b"\x00" * 4 + b"\x05"
    )
    secret = _shared_secret()
    packet = Packet()
    packet.header = (PAYLOAD_TYPE_RESPONSE << 2) | (
        ROUTE_TYPE_FLOOD if flood else ROUTE_TYPE_DIRECT
    )
    packet.path = bytearray(IN_PATH) if flood else bytearray()
    packet.path_len = IN_PATH_LEN if flood else 0
    packet.payload = bytearray(
        bytes([LOCAL_IDENTITY.get_public_key()[0], PEER_HASH])
        + CryptoUtils.encrypt_then_mac(secret[:16], secret, login_reply)
    )
    packet.payload_len = len(packet.payload)
    return packet


@pytest.mark.asyncio
async def test_login_response_handler_teaches_on_flood_login_response():
    """The forced-path regression: a DIRECT login is answered with a flood
    RESPONSE, and that is the only chance to teach before the first CLI REQ."""
    injector = _Injector()
    handler = LoginResponseHandler(LOCAL_IDENTITY, _contacts(), lambda _m: None)
    handler.set_packet_injector(injector)

    completions = []
    handler.register_login_callback(PEER_KEY, lambda success, data: completions.append(success))

    await handler(_login_reply_packet())
    await handler.return_path_teacher.wait_for_pending()

    assert completions == [True], "login response should still be delivered"
    assert len(injector.packets) == 1
    _, embedded_path = _decode_teach(injector.packets[0])
    assert embedded_path == IN_PATH


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def _core(contacts=None):
    return create_core_handlers(
        identity=LOCAL_IDENTITY,
        contacts=contacts if contacts is not None else _contacts(),
        channels=None,
        event_service=None,
        send_packet_fn=lambda *a, **k: None,
        log_fn=lambda _m: None,
        node_name="test",
    )


def test_factory_shares_one_teacher_between_response_handlers():
    """A shared instance keeps the per-contact cooldown and the evidence guard
    honest — two handlers each with their own teacher would double the transmit
    rate and lose the guard."""
    core = _core()
    assert core.protocol_response_handler.return_path_teacher is core.return_path_teacher
    assert core.login_response_handler.return_path_teacher is core.return_path_teacher


def test_set_packet_injector_also_wires_the_teacher():
    """Existing companion wiring calls only set_packet_injector; the teacher has
    to pick the transmit path up from there or it silently never fires."""
    core = _core()
    assert core.return_path_teacher.enabled is False
    core.protocol_response_handler.set_packet_injector(_Injector())
    assert core.return_path_teacher.enabled is True
    assert core.login_response_handler.return_path_teacher.enabled is True


def test_login_handler_can_wire_its_own_injector_standalone():
    handler = LoginResponseHandler(LOCAL_IDENTITY, _contacts(), lambda _m: None)
    assert handler.return_path_teacher.enabled is False
    handler.set_packet_injector(_Injector())
    assert handler.return_path_teacher.enabled is True


# --------------------------------------------------------------------------- #
# base_send retry hook
# --------------------------------------------------------------------------- #


class _StubSender(_SendOpsMixin):
    """Minimal carrier for the retry hook; only the handler accessor is used."""

    def __init__(self, handler):
        self._handler = handler

    def _get_protocol_response_handler(self):
        return self._handler


@pytest.mark.asyncio
async def test_retry_hook_asks_the_teacher_to_reverse_the_out_path():
    injector = _Injector()
    handler = ProtocolResponseHandler(lambda _m: None, LOCAL_IDENTITY, _contacts())
    handler.set_packet_injector(injector)

    await _StubSender(handler)._teach_return_path_before_retry(PEER_KEY, "REQ 0x01")
    await handler.return_path_teacher.wait_for_pending()

    assert len(injector.packets) == 1
    _, embedded_path = _decode_teach(injector.packets[0])
    assert embedded_path == bytes(reversed(OUT_PATH))


@pytest.mark.asyncio
async def test_retry_hook_is_best_effort_and_never_raises():
    """A failing teach must not abort the retry it precedes."""

    class _Boom:
        @property
        def return_path_teacher(self):
            raise RuntimeError("handler exploded")

    await _StubSender(_Boom())._teach_return_path_before_retry(PEER_KEY, "REQ")
    # No handler at all (companion subclasses may return None).
    await _StubSender(None)._teach_return_path_before_retry(PEER_KEY, "REQ")


@pytest.mark.asyncio
async def test_request_retry_loop_invokes_the_teach_hook():
    """Covers the wiring in _request_with_retries itself: without this, deleting
    the call would leave every other test in this file green."""
    calls = []

    class _Sender(_SendOpsMixin):
        async def _teach_return_path_before_retry(self, contact_pubkey, log_label):
            calls.append(bytes(contact_pubkey))

        def _apply_path_hash_mode(self, pkt):
            return None

        def _response_timeout_s(self, pkt, proxy):
            return 0.001

        async def _send_packet(self, pkt, wait_for_ack=False, expected_crc=None):
            return True

    def _build():
        pkt = Packet()
        pkt.header = (PAYLOAD_TYPE_RESPONSE << 2) | ROUTE_TYPE_DIRECT
        pkt.payload = bytearray(b"\x00" * 8)
        pkt.payload_len = 8
        pkt.path = bytearray()
        pkt.path_len = 0
        return pkt, None

    async def _always_timeout(_timeout):
        return {"timeout": True}

    result = await _Sender()._request_with_retries(
        _build, _always_timeout, object(), log_label="REQ", contact_pubkey=PEER_KEY
    )

    assert result["timeout"] is True
    # One teach before every attempt after the first.
    assert calls and all(c == PEER_KEY for c in calls)


@pytest.mark.asyncio
async def test_started_request_retry_loop_invokes_the_teach_hook():
    """Same coverage for the background continuation used by the frame-server
    API path (_finish_started_request)."""
    calls = []

    class _Sender(_SendOpsMixin):
        async def _teach_return_path_before_retry(self, contact_pubkey, log_label):
            calls.append(bytes(contact_pubkey))

        def _apply_path_hash_mode(self, pkt):
            return None

        def _response_timeout_s(self, pkt, proxy):
            return 0.001

        async def _send_packet(self, pkt, wait_for_ack=False, expected_crc=None):
            return True

    def _build():
        pkt = Packet()
        pkt.header = (PAYLOAD_TYPE_RESPONSE << 2) | ROUTE_TYPE_DIRECT
        pkt.payload = bytearray(b"\x00" * 8)
        pkt.payload_len = 8
        pkt.path = bytearray()
        pkt.path_len = 0
        return pkt, None

    async def _always_timeout(_timeout):
        return {"timeout": True}

    await _Sender()._finish_started_request(
        _build,
        _always_timeout,
        object(),
        first_timeout_s=0.001,
        deadline=None,
        log_label="REQ",
        cleanup=None,
        response_tag_registered=None,
        contact_pubkey=PEER_KEY,
    )

    assert calls and all(c == PEER_KEY for c in calls)
