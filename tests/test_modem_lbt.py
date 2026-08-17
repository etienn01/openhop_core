"""Tests for the USB/TCP modem Listen-Before-Talk path.

Mirrors the SPI path: LBT is bounded in TIME, not in attempts, so short
jittered retries run for the whole budget however long the occupation lasts.

Also covers ERR_CHANNEL_BUSY: with firmware auto-CAD enabled
(CMD_SET_AUTO_CAD) the modem refuses a busy channel instead of trampling a
neighbour. That refusal is LBT feedback, retried within the same time budget,
and the send fails without forcing when the modem keeps refusing.
"""

import struct
import time
from unittest.mock import AsyncMock

from openhop_core.hardware.protocol_constants import (
    CMD_RX_START,
    CMD_TX_REQUEST,
    ERR_CHANNEL_BUSY,
)
from openhop_core.hardware.tcp_radio import TCPLoRaRadio
from openhop_core.hardware.usb_radio import USBLoRaRadio

TX_DONE_PAYLOAD = struct.pack("<I", 12_000)  # 12 ms airtime


def _usb_radio(**kwargs) -> USBLoRaRadio:
    radio = USBLoRaRadio(
        port="/dev/null",
        lbt_max_wait_seconds=kwargs.pop("lbt_max_wait_seconds", 0.5),
        lbt_retry_interval_ms=kwargs.pop("lbt_retry_interval_ms", 20),
        **kwargs,
    )
    radio._initialized = True  # begin() never runs: transport is stubbed
    return radio


def _tcp_radio(**kwargs) -> TCPLoRaRadio:
    radio = TCPLoRaRadio(
        host="modem.invalid",
        lbt_max_wait_seconds=kwargs.pop("lbt_max_wait_seconds", 0.5),
        lbt_retry_interval_ms=kwargs.pop("lbt_retry_interval_ms", 20),
        **kwargs,
    )
    radio._initialized = True
    return radio


def _script_modem(radio, tx_responses):
    """Stub _send_command: consume one scripted entry per CMD_TX_REQUEST,
    answer every other command (e.g. CMD_RX_START) with success.

    An entry is either bytes (returned as the TX_DONE payload) or an error
    code (the modem answered CMD_ERROR: record it and return None, exactly
    like the real dispatch path does).
    """
    calls = {"tx": 0, "other": []}

    async def _impl(cmd, payload=b"", expect_cmd=None, timeout=None):
        if cmd == CMD_TX_REQUEST:
            calls["tx"] += 1
            entry = tx_responses.pop(0) if tx_responses else TX_DONE_PAYLOAD
            if isinstance(entry, bytes):
                return entry
            radio._last_modem_error = entry
            return None
        calls["other"].append(cmd)
        return b""

    radio._send_command = AsyncMock(side_effect=_impl)
    return calls


# ─── time-bounded host-side CAD loop ─────────────────────────────────


async def test_usb_lbt_budget_is_time_bounded():
    radio = _usb_radio()
    radio._perform_cad = AsyncMock(return_value=True)  # channel never clears
    _script_modem(radio, [TX_DONE_PAYLOAD])

    started = time.monotonic()
    result = await radio.send(b"payload")
    elapsed = time.monotonic() - started

    assert result is not None  # forced TX at budget exhaustion, loudly logged
    # Many short checks within the budget; the exact count is timing-dependent.
    assert radio._perform_cad.await_count > 5
    assert 0.4 <= elapsed <= 2.0
    assert sum(result["lbt_backoff_delays_ms"]) <= 600.0
    assert all(10.0 <= d <= 30.0 for d in result["lbt_backoff_delays_ms"])


async def test_usb_lbt_clear_channel_has_no_delays():
    radio = _usb_radio()
    radio._perform_cad = AsyncMock(return_value=False)
    _script_modem(radio, [TX_DONE_PAYLOAD])

    result = await radio.send(b"payload")

    assert result is not None
    assert result["lbt_backoff_delays_ms"] == []
    assert result["lbt_channel_busy"] is False


async def test_tcp_lbt_budget_is_time_bounded():
    radio = _tcp_radio()
    radio._perform_cad = AsyncMock(return_value=True)
    _script_modem(radio, [TX_DONE_PAYLOAD])

    started = time.monotonic()
    result = await radio.send(b"payload")
    elapsed = time.monotonic() - started

    assert result is not None
    assert radio._perform_cad.await_count > 5
    assert 0.4 <= elapsed <= 2.0


# ─── ERR_CHANNEL_BUSY: firmware auto-CAD refusal is LBT feedback ─────


async def test_usb_modem_busy_refusal_is_retried_within_budget():
    """The modem answering ERR_CHANNEL_BUSY to CMD_TX_REQUEST means its own
    auto-CAD saw a busy channel: retry within the LBT time budget instead of
    reporting a TX failure."""
    radio = _usb_radio(lbt_enabled=False)  # isolate the modem-side path
    calls = _script_modem(radio, [ERR_CHANNEL_BUSY, ERR_CHANNEL_BUSY, TX_DONE_PAYLOAD])

    result = await radio.send(b"payload")

    assert result is not None
    assert calls["tx"] == 3
    assert len(result["lbt_backoff_delays_ms"]) == 2  # one wait per refusal
    assert result["lbt_channel_busy"] is True
    assert CMD_RX_START in calls["other"]  # RX restored after success


async def test_usb_modem_busy_for_whole_budget_fails_without_forcing():
    """Unlike the host-side loop there is nothing to force through a modem
    that keeps refusing: it is alive and answering, not wedged. send()
    reports the failure once the budget is exhausted."""
    radio = _usb_radio(lbt_enabled=False)
    calls = _script_modem(radio, [ERR_CHANNEL_BUSY] * 1000)

    started = time.monotonic()
    result = await radio.send(b"payload")
    elapsed = time.monotonic() - started

    assert result is None
    assert 0.4 <= elapsed <= 2.0  # bounded by the budget, not the 1000 refusals
    assert 2 < calls["tx"] < 100


async def test_usb_non_busy_error_is_not_retried():
    radio = _usb_radio(lbt_enabled=False)
    calls = _script_modem(radio, [0x04])  # ERR_TX_TIMEOUT: a real failure

    result = await radio.send(b"payload")

    assert result is None
    assert calls["tx"] == 1


async def test_tcp_modem_busy_refusal_is_retried_within_budget():
    radio = _tcp_radio(lbt_enabled=False)
    calls = _script_modem(radio, [ERR_CHANNEL_BUSY, TX_DONE_PAYLOAD])

    result = await radio.send(b"payload")

    assert result is not None
    assert calls["tx"] == 2
    assert len(result["lbt_backoff_delays_ms"]) == 1


# ─── knobs ───────────────────────────────────────────────────────────


def test_lbt_knob_clamps_and_defaults():
    clamped = USBLoRaRadio(
        port="/dev/null", lbt_max_wait_seconds=0.0, lbt_retry_interval_ms=5
    )
    assert clamped.lbt_max_wait_seconds == 0.5
    assert clamped.lbt_retry_interval_ms == 20

    default_usb = USBLoRaRadio(port="/dev/null")
    default_tcp = TCPLoRaRadio(host="modem.invalid")
    for radio in (default_usb, default_tcp):
        # Matches MeshCore: getCADFailMaxDuration() 4 s, 200 ms retry.
        assert radio.lbt_max_wait_seconds == 4.0
        assert radio.lbt_retry_interval_ms == 200
