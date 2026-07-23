"""Reciprocal return-path teaching for flood-routed replies.

MeshCore servers do **not** derive a reply route by reversing the path an
inbound request arrived on. ``simple_repeater``'s ``MyMesh::onPeerDataRecv``
answers a DIRECT ``PAYLOAD_TYPE_REQ`` out of the per-client ``out_path`` stored
in its ACL::

    if (client->out_path_len != OUT_PATH_UNKNOWN) {
      sendDirect(reply, client->out_path, client->out_path_len, ...);
    } else {
      sendFloodReply(reply, ...);
    }

That ``out_path`` has exactly one writer — ``onPeerPathRecv``, i.e. receiving a
``PAYLOAD_TYPE_PATH`` packet from the client. A client that never sends one
leaves the server replying down whatever route it last stored, which after a
direct login is either nothing (``OUT_PATH_UNKNOWN`` -> flood) or a stale route
from an earlier session (``MyMesh.cpp``: ``if (is_flood) client->out_path_len =
OUT_PATH_UNKNOWN;`` — a *direct* login deliberately does not reset it).

MeshCore's chat client covers this with a self-healing retry
(``BaseChatMesh::onPeerDataRecv`` RESPONSE branch and ``onAckRecv``)::

    if (packet->isRouteFlood() && from.out_path_len != OUT_PATH_UNKNOWN) {
      // we have direct path, but other node is still sending flood response,
      // so maybe they didn't receive reciprocal path properly(?)
      handleReturnPathRetry(from, packet->path, packet->path_len);
    }

A flood reply from a peer we *already* hold a direct ``out_path`` for is the
tell: the peer has no usable route back to us. The fix is to re-teach it with a
``createPathReturn`` sent DIRECT along our own ``out_path``.

openHop previously sent a reciprocal PATH in only one place — on receiving a
flood ``PAYLOAD_TYPE_PATH`` (:mod:`.protocol_response`). That covers a *flood*
login, whose response comes back wrapped in a PATH. It does not cover a login
sent DIRECT (the user-forced-path case), which the server answers with a plain
flood ``RESPONSE``. This module supplies the missing parity.

Deliberate deviations from firmware, each documented at its definition:

* ``RETURN_PATH_COOLDOWN_S`` throttles re-teaching per contact. Firmware has no
  such throttle; openHop adds one so a burst of flood replies cannot turn into a
  burst of transmits.
* Firmware paces its retry with ``sendDirect(..., 3000)`` — a *queued* send with
  a 3 s delay, which never blocks packet processing. openHop's injector takes no
  delay argument, so a teach is dispatched as a background task instead. It must
  not be awaited inline on the RX path: the injector may block on a TX lock and
  an airtime budget, which would delay delivery of the very reply that triggered
  it, potentially past its own response timeout.
* ``maybe_teach_reverse_of_out_path`` has no firmware counterpart at all. See its
  docstring for the guard that keeps a guess from overwriting a known-good route.

Not yet covered: firmware also re-teaches from ``onAckRecv``. A discrete ACK on
the wire is a bare 4-byte CRC with no source hash, and openHop's dispatcher
tracks pending ACK CRCs in a plain set with no contact attribution, so there is
nothing to key the re-teach on. Wiring that up needs a CRC->contact map in the
send path; it is not required for the request/response flows this module fixes.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Optional, Set

from ...protocol import Identity, Packet
from ...protocol.constants import PH_ROUTE_MASK, ROUTE_TYPE_DIRECT
from ...protocol.packet_builder import PacketBuilder
from ...protocol.packet_utils import PathUtils

# Minimum gap between two return-path teaches aimed at the same contact.
# Firmware has no equivalent throttle: it re-teaches on every qualifying flood
# reply. openHop adds one as a safety valve so a peer that keeps flooding (for
# example because our teach itself is not getting through) cannot drive an
# unbounded transmit rate.
#
# This is shorter than a typical request retry cadence but not uniformly so:
# measured adaptive response timeouts (SF10/250k/4:5, 40 B) run ~2.4 s zero-hop,
# ~4.2 s one-hop, ~4.8 s flood and ~6.1 s at two hops, so for near contacts the
# second and third retry-driven teaches are suppressed. That is intended — the
# first teach after a timeout still goes out immediately, because a contact with
# no prior teach has no cooldown entry, and re-teaching the same guess every two
# seconds buys nothing.
RETURN_PATH_COOLDOWN_S = 5.0


def reverse_path(path: bytes, path_len_byte: int) -> Optional[bytes]:
    """Reverse a routing path hash-by-hash.

    A path is ``hash_count`` entries of ``hash_size`` bytes each, ordered from
    the sender outward. The route back is the same repeaters in the opposite
    order, so the reversal has to work on whole hashes — reversing the raw bytes
    would corrupt any path using 2- or 3-byte hashes.

    Returns ``None`` when ``path_len_byte`` is malformed or does not describe
    exactly ``path``.
    """
    if not PathUtils.is_valid_path_len(path_len_byte):
        return None
    hash_size = PathUtils.get_path_hash_size(path_len_byte)
    hash_count = PathUtils.get_path_hash_count(path_len_byte)
    if len(path) != hash_size * hash_count:
        return None
    hops = [path[i * hash_size : (i + 1) * hash_size] for i in range(hash_count)]
    return b"".join(reversed(hops))


class ReturnPathTeacher:
    """Sends ``PAYLOAD_TYPE_PATH`` teaches so a peer learns its route back to us.

    One instance is shared by the login and protocol-response handlers (see
    :func:`..registry.create_core_handlers`) so the cooldown is accounted per
    contact rather than per handler.
    """

    def __init__(
        self,
        log_fn: Callable[[str], None],
        local_identity: Any,
        contact_book: Any,
        *,
        cooldown_s: float = RETURN_PATH_COOLDOWN_S,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._log = log_fn
        self._local_identity = local_identity
        self._contact_book = contact_book
        self._cooldown_s = cooldown_s
        self._time = time_fn
        self._injector: Optional[Callable] = None
        # contact pubkey -> monotonic timestamp of the last teach dispatched.
        # Bounded by the contact book: keys only come from resolved contacts.
        self._last_taught: dict[bytes, float] = {}
        # Contacts we have taught from a real inbound path. Once a contact is in
        # here, the reverse-of-out_path guess is permanently disabled for it —
        # see maybe_teach_reverse_of_out_path.
        self._taught_from_evidence: Set[bytes] = set()
        # In-flight teach tasks, so callers (and tests) can await completion.
        self._pending_teaches: Set[asyncio.Task] = set()

    def set_injector(self, injector: Optional[Callable]) -> None:
        """Set the async callable used to transmit a teach packet."""
        self._injector = injector

    def note_evidence_teach(self, contact_pubkey: bytes) -> None:
        """Record that a return path was taught to ``contact_pubkey`` from a real
        inbound path by some *other* code path.

        :meth:`ProtocolResponseHandler._send_reciprocal_path` also teaches a
        return path (its flood-PATH reciprocal). It must report that here, or
        this teacher would believe the contact has never been taught and would
        let the reverse-of-out_path guess overwrite a route that is known good.
        """
        key = bytes(contact_pubkey)
        self._taught_from_evidence.add(key)
        self._last_taught[key] = self._time()

    async def wait_for_pending(self) -> None:
        """Await every in-flight teach. Intended for shutdown and for tests."""
        if self._pending_teaches:
            await asyncio.gather(*tuple(self._pending_teaches), return_exceptions=True)

    @property
    def enabled(self) -> bool:
        """True when a transmit path has been wired up."""
        return self._injector is not None

    # ------------------------------------------------------------------
    # Firmware parity: re-teach after a flood reply
    # ------------------------------------------------------------------

    async def maybe_teach_from_flood_reply(
        self,
        pkt: Packet,
        contact_pubkey: bytes,
        src_hash: int,
        *,
        reason: str,
        shared_secret: Optional[bytes] = None,
    ) -> bool:
        """Re-teach our return path when a flood reply arrives from a known route.

        Mirrors ``BaseChatMesh::onPeerDataRecv``'s RESPONSE branch: only fires
        when the reply is flood-routed *and* we already hold a direct
        ``out_path`` for the sender. The path embedded in the teach is the flood
        accumulation path the reply arrived on, which is precisely the route
        from the peer back to us.

        Returns True when a teach was transmitted.
        """
        if self._injector is None:
            return False
        if not pkt.is_route_flood():
            return False

        out_path, out_path_len = self._known_out_path(contact_pubkey)
        if out_path is None:
            # No stored route to the peer, so a flood reply is the expected
            # behaviour, not a symptom. Firmware's OUT_PATH_UNKNOWN guard.
            return False

        in_len_byte = int(getattr(pkt, "path_len", 0) or 0)
        if not PathUtils.is_valid_path_len(in_len_byte):
            self._log(
                f"[ReturnPath] Skip teach to 0x{src_hash:02X}: "
                f"inbound path_len 0x{in_len_byte:02X} is malformed"
            )
            return False
        in_byte_len = PathUtils.get_path_byte_len(in_len_byte)
        raw_path = bytes(pkt.path or b"")
        if len(raw_path) < in_byte_len:
            self._log(
                f"[ReturnPath] Skip teach to 0x{src_hash:02X}: inbound path is "
                f"{len(raw_path)}B, path_len byte declares {in_byte_len}B"
            )
            return False

        return await self._teach(
            contact_pubkey=contact_pubkey,
            dest_hash=src_hash,
            embedded_path=raw_path[:in_byte_len],
            embedded_len_byte=in_len_byte,
            out_path=out_path,
            out_path_len=out_path_len,
            shared_secret=shared_secret,
            reason=reason,
            from_evidence=True,
        )

    # ------------------------------------------------------------------
    # openHop hardening: re-teach when nothing comes back at all
    # ------------------------------------------------------------------

    async def maybe_teach_reverse_of_out_path(
        self,
        contact_pubkey: bytes,
        *,
        reason: str,
    ) -> bool:
        """Teach the reverse of our own ``out_path`` as the peer's route back.

        This has no firmware equivalent and covers the case firmware cannot:
        when the peer answers DIRECT down a stale route, *nothing* arrives, so
        there is no flood reply to trigger :meth:`maybe_teach_from_flood_reply`
        and no inbound path to embed. The best available guess is that the route
        is symmetric — the same repeaters traversed in reverse — which is
        exactly what a user-forced path asserts.

        It is only ever a guess, and the peer applies whatever it last received
        (``MyMesh::onPeerPathRecv`` overwrites ``client->out_path``
        unconditionally). So it is strictly a **last resort**: once we have
        taught this contact from a real inbound path, the guess is disabled for
        good. Without that guard a merely slow first attempt would replace a
        correct, evidence-derived route with a symmetry assumption — and if the
        assumption is wrong the peer then replies into a void, no flood reply
        ever arrives again, and nothing can re-teach it. That is strictly worse
        than the pre-existing behaviour, where an untaught peer falls back to
        flood replies that do reach us.

        Only meaningful when we already hold an ``out_path``; a contact still on
        flood needs no teach because flood replies reach us anyway.

        Returns True when a teach was dispatched.
        """
        if self._injector is None:
            return False

        key = bytes(contact_pubkey)
        if key in self._taught_from_evidence:
            self._log(
                f"[ReturnPath] Skip reverse teach to 0x{key[0]:02X} ({reason}): "
                "already taught from an observed inbound path"
            )
            return False

        out_path, out_path_len = self._known_out_path(contact_pubkey)
        if out_path is None:
            return False

        embedded = reverse_path(out_path, out_path_len)
        if embedded is None:
            self._log(
                "[ReturnPath] Skip reverse teach: stored out_path "
                f"({len(out_path)}B) does not match path_len 0x{out_path_len:02X}"
            )
            return False

        return await self._teach(
            contact_pubkey=contact_pubkey,
            dest_hash=contact_pubkey[0],
            embedded_path=embedded,
            embedded_len_byte=out_path_len,
            out_path=out_path,
            out_path_len=out_path_len,
            shared_secret=None,
            reason=reason,
            from_evidence=False,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _known_out_path(self, contact_pubkey: bytes) -> tuple[Optional[bytes], int]:
        """Return ``(out_path, encoded_len)`` for a contact, or ``(None, -1)``.

        ``out_path_len`` is the encoded path_len byte, with -1 standing in for
        firmware's ``OUT_PATH_UNKNOWN``. Zero is a *known* route (zero-hop
        direct neighbour) and must not be confused with unknown, so the value is
        range-checked rather than truth-tested.
        """
        try:
            contact = self._contact_book.get_by_key(contact_pubkey)
        except Exception as e:
            self._log(f"[ReturnPath] Contact lookup failed: {e}")
            return None, -1
        if contact is None:
            return None, -1

        raw_len = getattr(contact, "out_path_len", -1)
        try:
            out_path_len = -1 if raw_len is None else int(raw_len)
        except (TypeError, ValueError):
            return None, -1
        if out_path_len < 0 or not PathUtils.is_valid_path_len(out_path_len):
            return None, -1

        out_path = bytes(getattr(contact, "out_path", b"") or b"")
        expected = PathUtils.get_path_byte_len(out_path_len)
        if len(out_path) < expected:
            self._log(
                f"[ReturnPath] Stored out_path is {len(out_path)}B but path_len "
                f"0x{out_path_len:02X} declares {expected}B — not teaching"
            )
            return None, -1
        return out_path[:expected], out_path_len

    def _shared_secret_for(self, contact_pubkey: bytes) -> Optional[bytes]:
        """Derive the X25519 shared secret with ``contact_pubkey``."""
        try:
            return Identity(bytes(contact_pubkey)).calc_shared_secret(
                self._local_identity.get_private_key()
            )
        except Exception as e:
            self._log(f"[ReturnPath] Failed to derive shared secret: {e}")
            return None

    def _cooldown_active(self, key: bytes) -> bool:
        last = self._last_taught.get(key)
        return last is not None and (self._time() - last) < self._cooldown_s

    async def _teach(
        self,
        *,
        contact_pubkey: bytes,
        dest_hash: int,
        embedded_path: bytes,
        embedded_len_byte: int,
        out_path: bytes,
        out_path_len: int,
        shared_secret: Optional[bytes],
        reason: str,
        from_evidence: bool,
    ) -> bool:
        """Build one return-path teach and dispatch it. True when dispatched.

        The cooldown is claimed *synchronously*, before anything is awaited, so
        two concurrent triggers for the same contact cannot both pass the check
        and both transmit. It is released again if the packet never reaches the
        injector, so a failed teach is retried on the next trigger instead of
        being muted for the whole cooldown window.
        """
        key = bytes(contact_pubkey)
        if self._cooldown_active(key):
            self._log(
                f"[ReturnPath] Skip teach to 0x{dest_hash:02X} ({reason}): "
                f"within {self._cooldown_s:.0f}s cooldown"
            )
            return False

        secret = shared_secret if shared_secret is not None else self._shared_secret_for(key)
        if not secret:
            return False

        try:
            teach = PacketBuilder.create_path_return(
                dest_hash=dest_hash,
                src_hash=self._local_identity.get_public_key()[0],
                secret=secret,
                path=embedded_path,
                extra_type=0xFF,  # no extra payload (firmware passes NULL/0)
                extra=b"",
                path_len_encoded=embedded_len_byte,
            )
            # createPathReturn yields a FLOOD PATH; firmware's handleReturnPathRetry
            # sends it DIRECT along the out_path we already hold for the peer.
            teach.header = (teach.header & ~PH_ROUTE_MASK) | ROUTE_TYPE_DIRECT
            teach.set_path(out_path, out_path_len)
        except Exception as e:
            self._log(f"[ReturnPath] Failed to build teach for 0x{dest_hash:02X}: {e}")
            return False

        # Claim the cooldown now: everything past this point is asynchronous.
        previous = self._last_taught.get(key)
        self._last_taught[key] = self._time()

        # Firmware queues this send (sendDirect with a 3s delay) rather than
        # blocking. Awaiting the injector here would stall the RX path — it can
        # wait on a TX lock and an airtime budget — and delay delivery of the
        # reply that triggered the teach.
        task = asyncio.ensure_future(
            self._dispatch_teach(
                teach=teach,
                key=key,
                previous_cooldown=previous,
                dest_hash=dest_hash,
                embedded_path=embedded_path,
                embedded_len_byte=embedded_len_byte,
                out_path=out_path,
                reason=reason,
                from_evidence=from_evidence,
            )
        )
        self._pending_teaches.add(task)
        task.add_done_callback(self._pending_teaches.discard)
        return True

    async def _dispatch_teach(
        self,
        *,
        teach: Packet,
        key: bytes,
        previous_cooldown: Optional[float],
        dest_hash: int,
        embedded_path: bytes,
        embedded_len_byte: int,
        out_path: bytes,
        reason: str,
        from_evidence: bool,
    ) -> None:
        """Transmit a prepared teach, rolling back the cooldown on failure."""
        try:
            await self._injector(teach)
        except Exception as e:
            if previous_cooldown is None:
                self._last_taught.pop(key, None)
            else:
                self._last_taught[key] = previous_cooldown
            self._log(f"[ReturnPath] Failed to send teach to 0x{dest_hash:02X}: {e}")
            return

        # Only a confirmed send counts as evidence: marking it optimistically
        # would permanently disable the reverse-path fallback on a teach that
        # never left the node.
        if from_evidence:
            self._taught_from_evidence.add(key)

        self._log(
            f"[ReturnPath] Taught 0x{dest_hash:02X} its route back ({reason}): "
            f"embedded={embedded_path.hex() or '(zero-hop)'} "
            f"(path_len=0x{embedded_len_byte:02X}), "
            f"sent DIRECT via {out_path.hex() or '(zero-hop)'}"
        )
