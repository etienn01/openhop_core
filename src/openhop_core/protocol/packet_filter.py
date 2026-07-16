"""
Simple packet filter for dispatcher-level routing decisions.

This handles only the essential routing concerns:
- Duplicate detection
- Packet blacklisting for malformed packets
- Basic packet hash tracking
"""

import hashlib
import time
from collections import OrderedDict
from typing import Dict, Set


class PacketFilter:
    """Lightweight packet filter for dispatcher routing decisions."""

    def __init__(self, window_seconds: int = 30):
        self.window_seconds = window_seconds
        self._packet_hashes: Dict[str, float] = {}  # packet_hash -> timestamp
        self._blacklist: Set[str] = set()  # blacklisted packet hashes

    def generate_hash(self, data: bytes) -> str:
        """Generate a hash for packet data."""
        return hashlib.sha256(data).hexdigest()[:16]

    def is_duplicate(self, packet_hash: str) -> bool:
        """Check if we've seen this packet recently."""
        now = time.time()
        if packet_hash in self._packet_hashes:
            age = now - self._packet_hashes[packet_hash]
            if age < self.window_seconds:
                return True
        return False

    def track_packet(self, packet_hash: str) -> None:
        """Track a packet hash with current timestamp."""
        self._packet_hashes[packet_hash] = time.time()

    def blacklist(self, packet_hash: str) -> None:
        """Add a packet hash to the blacklist."""
        self._blacklist.add(packet_hash)

    def is_blacklisted(self, packet_hash: str) -> bool:
        """Check if a packet hash is blacklisted."""
        return packet_hash in self._blacklist

    def cleanup_old_hashes(self) -> None:
        """Clean up old packet hashes beyond the deduplication window."""
        current_time = time.time()
        old_hashes = [
            h for h, ts in self._packet_hashes.items() if current_time - ts > self.window_seconds
        ]
        for h in old_hashes:
            del self._packet_hashes[h]

    def get_stats(self) -> dict:
        """Get basic filter statistics."""
        return {
            "tracked_packets": len(self._packet_hashes),
            "blacklisted_packets": len(self._blacklist),
            "window_seconds": self.window_seconds,
        }

    def clear(self) -> None:
        """Clear all tracked data."""
        self._packet_hashes.clear()
        self._blacklist.clear()


class PacketHashCache:
    """Bounded TTL cache for full packet-hash keys.

    This is intended for application-level message de-duplication, where a
    full hash avoids treating two distinct packets as the same message.
    """

    def __init__(self, ttl_seconds: float = 60.0, max_entries: int = 4096):
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: OrderedDict[str, float] = OrderedDict()

    def _evict_expired(self, now: float) -> None:
        while self._entries:
            _, seen_at = next(iter(self._entries.items()))
            if now - seen_at <= self.ttl_seconds:
                break
            self._entries.popitem(last=False)

    def check_and_add(self, packet_hash: str) -> bool:
        """Return whether *packet_hash* is still cached, otherwise store it.

        A hit refreshes the entry, so suppression of a key extends while
        duplicates keep arriving; an entry only expires after a full quiet
        TTL. MeshCore's seen table has no expiry at all (a cyclic buffer of
        160 hashes displaced by newer traffic), so refreshing keeps this
        bounded cache closer to firmware behavior than a fixed window.
        """
        now = time.monotonic()
        self._evict_expired(now)
        hit = packet_hash in self._entries
        self._entries[packet_hash] = now
        self._entries.move_to_end(packet_hash)
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return hit
