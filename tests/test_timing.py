"""Tests for adaptive request timeouts (companion/timing.py)."""

import math

from pymc_core.companion import timing
from pymc_core.protocol.packet_utils import PathUtils


def test_estimate_airtime_matches_semtech_formula():
    """SF10/250kHz/CR4-5, 24-byte packet ~= 185ms (hand-computed Semtech airtime)."""
    air = timing.estimate_airtime_ms(24, sf=10, bw_hz=250000, cr=1)
    assert math.isclose(air, 185.3, rel_tol=0.02)


def test_airtime_grows_with_spreading_factor():
    """Higher SF => much longer airtime (each +1 SF roughly doubles symbol time)."""
    a_sf8 = timing.estimate_airtime_ms(40, sf=8, bw_hz=250000, cr=1)
    a_sf10 = timing.estimate_airtime_ms(40, sf=10, bw_hz=250000, cr=1)
    a_sf12 = timing.estimate_airtime_ms(40, sf=12, bw_hz=125000, cr=1)
    assert a_sf8 < a_sf10 < a_sf12


def test_flood_timeout_matches_firmware_formula():
    assert timing.calc_flood_timeout_ms(200.0) == int(500 + 16.0 * 200.0)


def test_direct_timeout_uses_hop_count_not_raw_byte():
    # 0x42 encodes hash_size=2, hop_count=2 -> firmware (hops+1) factor of 3.
    out_path_len = 0x42
    assert PathUtils.get_path_hash_count(out_path_len) == 2
    expected = int(500 + (6.0 * 200.0 + 250) * (2 + 1))
    assert timing.calc_direct_timeout_ms(200.0, out_path_len) == expected


def test_direct_timeout_zero_hop():
    # 0x40 -> hash_size 2, 0 hops -> factor (0+1).
    expected = int(500 + (6.0 * 200.0 + 250) * 1)
    assert timing.calc_direct_timeout_ms(200.0, 0x40) == expected


def test_response_timeout_is_clamped():
    # Tiny airtime (fast SF, small packet) must not drop below the floor.
    fast = timing.response_timeout_ms(
        raw_length=12, is_flood=False, out_path_len=0, sf=7, bw_hz=500000, cr=1
    )
    assert fast == timing.MIN_TIMEOUT_MILLIS
    # Huge multi-hop flood must not exceed the ceiling.
    slow = timing.response_timeout_ms(
        raw_length=200, is_flood=True, out_path_len=0, sf=12, bw_hz=125000, cr=4
    )
    assert slow == timing.MAX_TIMEOUT_MILLIS


def test_flood_vs_direct_selection():
    """A typical small request lands in a sane few-second window for both routes."""
    flood = timing.response_timeout_ms(
        raw_length=53, is_flood=True, out_path_len=-1, sf=10, bw_hz=250000, cr=1
    )
    direct = timing.response_timeout_ms(
        raw_length=22, is_flood=False, out_path_len=0x42, sf=10, bw_hz=250000, cr=1
    )
    assert timing.MIN_TIMEOUT_MILLIS <= flood <= timing.MAX_TIMEOUT_MILLIS
    assert timing.MIN_TIMEOUT_MILLIS <= direct <= timing.MAX_TIMEOUT_MILLIS
