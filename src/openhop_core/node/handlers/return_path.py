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
* Firmware paces its retry with ``sendDirect(..., 3000)`` — a *queued* send whose
  3 s delay is a settle window (``queueOutbound(..., futureMillis(3000))``), not a
  busy wait. openHop honours that window (``RETURN_PATH_SETTLE_S``) in a background
  task rather than inline on the RX path: the injector may block on a TX lock and
  an airtime budget, and awaiting it inline would delay delivery of the very reply
  that triggered the teach, potentially past its own response timeout.
* openHop picks the teach source by a hop-penalized last-hop RSSI across every
  copy of the flood reply, collected pre-dedup via ``note_flood_copy``. Firmware
  embeds whichever copy it processed first, and only its (SNR-scored,
  ships-disabled) reception hold makes that the best one; neither firmware mode
  weighs hop count, so a strong nearby repeater can append itself as an extra
  hop. openHop does not lean on the hold and discounts extra hops. See
  ``RETURN_PATH_SETTLE_S`` and ``RETURN_PATH_HOP_PENALTY_DB``. There is
  deliberately no cross-reply downgrade guard:
  firmware has none, and a flood reply from a peer we hold an ``out_path`` for is
  by this module's own premise a broken return route, so refusing to re-teach it
  because a newer copy is weaker than a previous (non-working) one would prolong
  the failure.
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
from collections import OrderedDict
from typing import Any, Callable, Optional, Set

from ...protocol import Identity, Packet
from ...protocol.constants import PAYLOAD_TYPE_RESPONSE, PH_ROUTE_MASK, ROUTE_TYPE_DIRECT
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

# How long to keep collecting copies of a flood reply before teaching from the
# best-received one. Firmware sends its retry with ``sendDirect(..., 3000)``;
# that 3 s is a queued-send "not before" schedule (``queueOutbound(...,
# futureMillis(3000))``) — a settle window, not just pacing.
#
# A flood reply reaches us by several routes, and openHop's dedup, like
# firmware's ``hasSeen``, is "first packet wins". The first copy to arrive is
# NOT the best route -- observed on a live SF7/62.5k mesh, four copies of one
# login reply landed over ~6 s at -102, -62, -41 and -24 dBm, weakest first.
# Teaching from the first copy tells the peer to answer down its most marginal
# link, and its replies then die there while flood replies still get through,
# which looks exactly like the bug this module was written to fix.
#
# Firmware avoids this only when its reception-quality hold is enabled: the hold
# reorders flood processing so the best-scored copy is seen (and marks-seen)
# first, making firmware's embedded ``packet->path`` the best copy. But the hold
# is scored on SNR alone and ships DISABLED (``rx_delay_base`` is memset-zeroed;
# the enabling line is commented "enable once new algo fixed"), so on a stock
# node firmware embeds the first-arrived copy -- and its 3 s delay only defers
# the transmit, it does not re-pick the copy. openHop therefore selects the
# teach source itself, by a hop-penalized last-hop RSSI (see
# ``RETURN_PATH_HOP_PENALTY_DB`` and ``note_flood_copy``), during this window
# instead of leaning on the hold. Unchanged on the wire, strictly better than
# firmware when the hold is off and no worse when it is on.
RETURN_PATH_SETTLE_S = 3.0

# Cost, in dB of last-hop RSSI, charged per hop when choosing between copies of a
# flood reply. Copies are ranked by ``rssi - HOP_PENALTY_DB * hop_count`` so a
# longer route only wins when it is *materially* stronger, not merely a couple dB.
#
# Neither firmware mode weighs hop count: the default embeds the first-arrived
# copy (Mesh.cpp notes the path "may not be the best in terms of hops"), and the
# optional SNR hold would embed the strongest copy regardless of length. Without
# this penalty, a strong nearby repeater -- e.g. one on the same desk as the
# node -- re-floods almost every reply and appends itself as a final hop, so
# pure best-RSSI would route returns through it even when a shorter route is
# adequate. ~10 dB/hop keeps that extra hop only when it genuinely buys signal:
# a 3-hop copy at -13 still beats a 2-hop copy at -40, but loses to one at -15.
RETURN_PATH_HOP_PENALTY_DB = 10.0

# Bounds for the best-copy table (``note_flood_copy``): entries older than the
# TTL are dropped and the table never holds more than ``_MAX`` hashes (oldest
# evicted). The TTL only has to outlive one settle window; it is clamped up to
# cover a custom ``settle_s`` in __init__.
RETURN_PATH_COPY_TTL_S = 8.0
RETURN_PATH_COPY_CACHE_MAX = 128


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


class _FloodCopy:
    """The best-received copy of one flood reply, keyed by its packet hash.

    ``rssi`` is the last-hop RSSI of the copy (higher is stronger); ``path`` /
    ``len_byte`` are the flood accumulation path that copy arrived on — the route
    the peer should be taught to use back to us. ``score`` is the hop-penalized
    rank (``rssi - HOP_PENALTY_DB * hop_count``) copies are compared on, so a
    shorter route is preferred unless a longer one is materially stronger.
    ``ts`` is the monotonic time the entry was last improved, for TTL pruning.
    """

    __slots__ = ("path", "len_byte", "rssi", "score", "ts")

    def __init__(self, path: bytes, len_byte: int, rssi: int, score: float, ts: float) -> None:
        self.path = path
        self.len_byte = len_byte
        self.rssi = rssi
        self.score = score
        self.ts = ts


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
        settle_s: float = RETURN_PATH_SETTLE_S,
        hop_penalty_db: float = RETURN_PATH_HOP_PENALTY_DB,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._log = log_fn
        self._local_identity = local_identity
        self._contact_book = contact_book
        self._cooldown_s = cooldown_s
        self._settle_s = settle_s
        self._hop_penalty_db = hop_penalty_db
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
        # packet hash -> best-received copy, populated pre-dedup by
        # note_flood_copy across the settle window. Bounded and TTL-pruned; the
        # TTL must outlive one settle window.
        self._recent_copies: "OrderedDict[bytes, _FloodCopy]" = OrderedDict()
        self._copy_ttl_s = max(RETURN_PATH_COPY_TTL_S, self._settle_s + 2.0)
        self._copy_cache_max = RETURN_PATH_COPY_CACHE_MAX
        # Cached destination hash (our own) used to filter recorded copies to
        # replies addressed to us. Resolved lazily so a late-bound identity still
        # works; None means "record regardless" rather than dropping everything.
        self._local_hash: Optional[int] = None

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

    def note_flood_copy(self, pkt: Packet, data: Any = None, analysis: Any = None) -> None:
        """Record a flood reply copy for best-route selection (raw RX subscriber).

        Wired via ``dispatcher.add_raw_packet_subscriber``, which fires for every
        received packet *before* dedup — the only place later copies of a flood
        reply are visible, since dedup drops them before any handler runs. Keeps,
        per packet hash, the highest-scoring copy (hop-penalized last-hop RSSI,
        see :meth:`_copy_score`) so a teach triggered on the first-arrived copy
        can still embed the best route (see :meth:`maybe_teach_from_flood_reply`).

        Best-effort and cheap: only flood ``RESPONSE`` packets addressed to us are
        recorded (exactly what triggers a teach), and any error is swallowed —
        this runs on the hot RX path and must never disturb packet processing.
        """
        if self._injector is None:
            return
        try:
            if not pkt.is_route_flood():
                return
            if pkt.get_payload_type() != PAYLOAD_TYPE_RESPONSE:
                return
            payload = getattr(pkt, "payload", None)
            if payload is None or len(payload) < 2:
                return
            if not self._addressed_to_us(payload[0]):
                return
            len_byte = int(getattr(pkt, "path_len", 0) or 0)
            path = self._validated_inbound_path(pkt, len_byte)
            if path is None:
                return
            packet_hash = bytes(pkt.calculate_packet_hash())
            rssi = int(getattr(pkt, "_rssi", 0) or 0)
            self._record_copy(packet_hash, path, len_byte, rssi)
        except Exception as e:  # never let a bad copy disturb RX
            self._log(f"[ReturnPath] note_flood_copy ignored a packet: {e}")

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

        The triggering ``pkt`` is only the *first-arrived* copy (dedup drops the
        rest before this handler runs). This copy is used as a seed/fallback, but
        the teach is delayed by :data:`RETURN_PATH_SETTLE_S` and then embeds the
        best-RSSI copy collected pre-dedup by :meth:`note_flood_copy` — see the
        constant's docstring for why first-arrived is often the worst route.

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
        embedded_path = self._validated_inbound_path(pkt, in_len_byte)
        if embedded_path is None:
            self._log(
                f"[ReturnPath] Skip teach to 0x{src_hash:02X}: inbound path "
                f"(path_len 0x{in_len_byte:02X}) is malformed or truncated"
            )
            return False

        # Seed the best-copy table with this first-arrived copy so a teach still
        # has a source even when no raw subscriber is wired (standalone/tests);
        # note_flood_copy improves it with better copies during the settle window.
        packet_hash = bytes(pkt.calculate_packet_hash())
        self._record_copy(
            packet_hash, embedded_path, in_len_byte, int(getattr(pkt, "_rssi", 0) or 0)
        )

        return await self._teach(
            contact_pubkey=contact_pubkey,
            dest_hash=src_hash,
            embedded_path=embedded_path,
            embedded_len_byte=in_len_byte,
            out_path=out_path,
            out_path_len=out_path_len,
            shared_secret=shared_secret,
            reason=reason,
            from_evidence=True,
            hash_key=packet_hash,
            settle_s=self._settle_s,
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

    # ---- best-copy collection (hop-penalized RSSI selection) ---------------

    def _copy_score(self, rssi: int, len_byte: int) -> float:
        """Rank a flood copy: last-hop RSSI minus a per-hop penalty.

        Prefers a shorter return route unless a longer one is materially
        stronger. See :data:`RETURN_PATH_HOP_PENALTY_DB`.
        """
        try:
            hops = PathUtils.get_path_hash_count(len_byte)
        except Exception:
            hops = 0
        return rssi - self._hop_penalty_db * hops

    def _addressed_to_us(self, dest_hash: int) -> bool:
        """True when ``dest_hash`` is our own hash (or our hash is unknown).

        Filters the pre-dedup firehose down to replies actually meant for us.
        If our hash cannot be resolved we record regardless rather than drop
        everything — a slightly larger table is safer than a silent no-op.
        """
        if self._local_hash is None:
            try:
                self._local_hash = self._local_identity.get_public_key()[0]
            except Exception:
                return True
        return dest_hash == self._local_hash

    def _validated_inbound_path(self, pkt: Packet, len_byte: int) -> Optional[bytes]:
        """The inbound flood path trimmed to its declared length, or None.

        Shared by :meth:`note_flood_copy` (hot path) and
        :meth:`maybe_teach_from_flood_reply`.
        """
        if not PathUtils.is_valid_path_len(len_byte):
            return None
        byte_len = PathUtils.get_path_byte_len(len_byte)
        raw = bytes(getattr(pkt, "path", b"") or b"")
        if len(raw) < byte_len:
            return None
        return raw[:byte_len]

    def _record_copy(self, packet_hash: bytes, path: bytes, len_byte: int, rssi: int) -> None:
        """Keep the highest-scoring copy of a flood reply, keyed by packet hash.

        Copies are ranked by :meth:`_copy_score` (hop-penalized RSSI), so a
        stronger-but-longer route only displaces a shorter one when it wins on
        score, not on raw RSSI alone.
        """
        now = self._time()
        self._prune_copies(now)
        score = self._copy_score(rssi, len_byte)
        existing = self._recent_copies.get(packet_hash)
        if existing is not None and score <= existing.score:
            existing.ts = now  # refresh so the winning copy outlives the window
            self._recent_copies.move_to_end(packet_hash)
            return
        self._recent_copies[packet_hash] = _FloodCopy(path, len_byte, rssi, score, now)
        self._recent_copies.move_to_end(packet_hash)
        if len(self._recent_copies) > self._copy_cache_max:
            self._recent_copies.popitem(last=False)  # evict least-recently-touched

    def _prune_copies(self, now: float) -> None:
        """Drop copies older than the TTL. Front entries are the oldest-touched."""
        ttl = self._copy_ttl_s
        while self._recent_copies:
            oldest = next(iter(self._recent_copies))
            if (now - self._recent_copies[oldest].ts) <= ttl:
                break
            self._recent_copies.popitem(last=False)

    async def _settle(self, seconds: float) -> None:
        """Wait out the teach settle window (overridable by tests/subclasses)."""
        await asyncio.sleep(seconds)

    # ---- teach dispatch ---------------------------------------------------

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
        hash_key: Optional[bytes] = None,
        settle_s: float = 0.0,
    ) -> bool:
        """Schedule one return-path teach. True when a teach was scheduled.

        The cooldown is claimed *synchronously*, before anything is awaited, so
        two concurrent triggers for the same contact cannot both pass the check
        and both transmit. It is released again if the packet never reaches the
        injector, so a failed teach is retried on the next trigger instead of
        being muted for the whole cooldown window.

        The packet is *not* built here: with a ``settle_s`` window the best copy
        is not known until the window closes, so build and transmit both happen
        in :meth:`_dispatch_teach`. The shared secret is derived up front so a
        contact that cannot be resolved fails synchronously (returns False)
        rather than after the settle delay.
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

        # Claim the cooldown now: everything past this point is asynchronous.
        previous = self._last_taught.get(key)
        self._last_taught[key] = self._time()

        # Firmware queues this send (sendDirect with a 3s delay) rather than
        # blocking. Awaiting it here would stall the RX path — the injector can
        # wait on a TX lock and an airtime budget — and delay delivery of the
        # reply that triggered the teach.
        task = asyncio.ensure_future(
            self._dispatch_teach(
                key=key,
                previous_cooldown=previous,
                dest_hash=dest_hash,
                secret=secret,
                embedded_path=embedded_path,
                embedded_len_byte=embedded_len_byte,
                out_path=out_path,
                out_path_len=out_path_len,
                reason=reason,
                from_evidence=from_evidence,
                hash_key=hash_key,
                settle_s=settle_s,
            )
        )
        self._pending_teaches.add(task)
        task.add_done_callback(self._pending_teaches.discard)
        return True

    async def _dispatch_teach(
        self,
        *,
        key: bytes,
        previous_cooldown: Optional[float],
        dest_hash: int,
        secret: bytes,
        embedded_path: bytes,
        embedded_len_byte: int,
        out_path: bytes,
        out_path_len: int,
        reason: str,
        from_evidence: bool,
        hash_key: Optional[bytes],
        settle_s: float,
    ) -> None:
        """Wait out the settle window, then build and transmit the best teach.

        Rolls the cooldown back on any failure so the next trigger can retry.
        """
        if settle_s > 0.0:
            await self._settle(settle_s)

        # Pick the best-received copy collected during the window. Falls back to
        # the seed passed in (the first-arrived copy) when nothing better was
        # recorded, when the entry was evicted, or for a reverse-of-out_path
        # guess (which has no hash_key and no inbound copies to collect).
        chosen_rssi: Optional[int] = None
        if hash_key is not None:
            best = self._recent_copies.pop(hash_key, None)
            if best is not None:
                embedded_path = best.path
                embedded_len_byte = best.len_byte
                chosen_rssi = best.rssi

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

        if chosen_rssi is not None:
            try:
                hops = PathUtils.get_path_hash_count(embedded_len_byte)
            except Exception:
                hops = "?"
            rssi_note = f", best last-hop RSSI {chosen_rssi} over {hops} hop(s)"
        else:
            rssi_note = ""
        self._log(
            f"[ReturnPath] Taught 0x{dest_hash:02X} its route back ({reason}): "
            f"embedded={embedded_path.hex() or '(zero-hop)'} "
            f"(path_len=0x{embedded_len_byte:02X}){rssi_note}, "
            f"sent DIRECT via {out_path.hex() or '(zero-hop)'}"
        )
