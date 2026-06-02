"""Adaptive request timeouts mirroring MeshCore firmware.

The firmware companion (``BaseChatMesh``) sizes each request's response timeout
from the packet's airtime and route, then retries on that cadence:

    calcFloodTimeoutMillisFor(t)        = 500 + 16.0 * t
    calcDirectTimeoutMillisFor(t, hops) = 500 + (6.0 * t + 250) * (hops + 1)

where ``t`` is the estimated airtime in ms (see ``examples/companion_radio``).
pyMC previously used fixed 10 s / 15 s waits with no resend, so a single lost
packet stalled for 10-15 s where firmware recovers in ~3 s.  This module
reproduces the firmware math so login/stats/discovery use the same cadence.
"""

import math

from ..protocol.packet_utils import PathUtils

# Firmware constants (examples/companion_radio/MyMesh.cpp).
SEND_TIMEOUT_BASE_MILLIS = 500
FLOOD_SEND_TIMEOUT_FACTOR = 16.0
DIRECT_SEND_PERHOP_FACTOR = 6.0
DIRECT_SEND_PERHOP_EXTRA_MILLIS = 250

# Default number of attempts (initial send + resends) for a request before
# giving up. Each attempt waits one adaptive timeout. Firmware relies on the
# host app to re-issue; we resend internally so recovery is independent of it.
DEFAULT_MAX_ATTEMPTS = 3

# Guard rails so a fast SF doesn't produce a pathologically short timeout and a
# slow SF / huge path doesn't block for too long before a resend.
MIN_TIMEOUT_MILLIS = 1500
MAX_TIMEOUT_MILLIS = 12000


def estimate_airtime_ms(
    packet_length: int,
    sf: int,
    bw_hz: int,
    cr: int,
    preamble_symbols: int = 8,
    low_dr_opt: bool = None,
) -> float:
    """Estimate LoRa airtime (ms) for a packet, per the Semtech formula.

    Mirrors ``SX1262Wrapper`` airtime math: explicit header, CRC on. ``cr`` is
    the MeshCore coding-rate index (1->4/5 .. 4->4/8). ``packet_length`` is the
    full on-air byte length (use ``Packet.get_raw_length()``).
    """
    sf = max(6, min(12, int(sf)))
    bw_hz = int(bw_hz) or 250000
    cr = max(1, min(4, int(cr)))
    if low_dr_opt is None:
        low_dr_opt = sf >= 11 and bw_hz <= 125000
    ldro = 1 if low_dr_opt else 0

    symbol_time = (1 << sf) / float(bw_hz)
    preamble_time = (preamble_symbols + 4.25) * symbol_time
    tmp = 8 * packet_length - 4 * sf + 28 + 16 * 1 - 20 * 0  # crc=1, explicit header
    denom = 4 * (sf - 2 * ldro)
    if tmp > 0 and denom > 0:
        payload_symbols = 8 + max(math.ceil(tmp / denom) * (cr + 4), 0)
    else:
        payload_symbols = 8
    payload_time = payload_symbols * symbol_time
    return (preamble_time + payload_time) * 1000.0


def calc_flood_timeout_ms(airtime_ms: float) -> int:
    """Firmware ``calcFloodTimeoutMillisFor``."""
    return int(SEND_TIMEOUT_BASE_MILLIS + FLOOD_SEND_TIMEOUT_FACTOR * airtime_ms)


def calc_direct_timeout_ms(airtime_ms: float, out_path_len: int) -> int:
    """Firmware ``calcDirectTimeoutMillisFor`` (out_path_len is the encoded byte)."""
    hops = PathUtils.get_path_hash_count(out_path_len) if out_path_len > 0 else 0
    return int(
        SEND_TIMEOUT_BASE_MILLIS
        + (DIRECT_SEND_PERHOP_FACTOR * airtime_ms + DIRECT_SEND_PERHOP_EXTRA_MILLIS) * (hops + 1)
    )


def response_timeout_ms(
    raw_length: int,
    is_flood: bool,
    out_path_len: int,
    sf: int,
    bw_hz: int,
    cr: int,
    preamble_symbols: int = 8,
) -> int:
    """Adaptive response timeout (ms) for a request packet, clamped.

    ``is_flood`` selects the flood vs direct firmware formula; ``out_path_len``
    is the contact's encoded path_len byte (used for the per-hop direct term).
    """
    airtime = estimate_airtime_ms(raw_length, sf, bw_hz, cr, preamble_symbols)
    if is_flood:
        ms = calc_flood_timeout_ms(airtime)
    else:
        ms = calc_direct_timeout_ms(airtime, out_path_len)
    return max(MIN_TIMEOUT_MILLIS, min(MAX_TIMEOUT_MILLIS, ms))
