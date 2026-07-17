"""TX airtime duty-cycle budget in the Core dispatcher (client-repeat only).

Reproduces MeshCore Dispatcher.cpp budget mechanics: a leaky bucket that
refills at duty_cycle = 1/(1+airtime_factor) over a 1-hour window, is spent on
each transmit's estimated airtime, gates sends below est_airtime(255)/2, and
DELAYS (never drops) when short. The bucket is consulted only while
client-repeat is enabled; when disabled the send path is untouched.
"""

import asyncio

import pytest

from openhop_core.node import dispatcher as disp_mod
from openhop_core.node.dispatcher import DUTY_CYCLE_WINDOW_MS, MIN_TX_BUDGET_RESERVE_MS, Dispatcher
from openhop_core.protocol import Packet
from openhop_core.protocol.constants import PAYLOAD_TYPE_TXT_MSG, ROUTE_TYPE_FLOOD

SELF_KEY = b"0123456789abcdef0123456789abcdef"


class Radio:
    def __init__(self):
        self.rx_callback = None
        self.send_count = 0
        self.tx_data = None

    def set_rx_callback(self, cb):
        self.rx_callback = cb

    async def send(self, data):
        self.send_count += 1
        self.tx_data = data
        return {"ok": 1}

    def get_last_rssi(self):
        return -70

    def get_last_snr(self):
        return 8.0


class Identity:
    def get_public_key(self):
        return SELF_KEY


class Clock:
    """Virtual monotonic clock; its sleep advances time instead of blocking."""

    def __init__(self, start=1000.0):
        self.t = start
        self.slept = []

    def monotonic(self):
        return self.t

    async def sleep(self, secs):
        self.slept.append(secs)
        if secs > 0:
            self.t += secs


def _make(factor=1.0, enabled=True):
    d = Dispatcher(Radio(), dedupe_enabled=True)
    d.local_identity = Identity()
    d.airtime_budget_factor = factor
    if enabled:
        d.set_client_repeat_enabled(True)
    return d, d.radio


def _flood_txt():
    p = Packet()
    p.header = (PAYLOAD_TYPE_TXT_MSG << 2) | ROUTE_TYPE_FLOOD
    p.payload = bytearray([0x77, 0x99]) + bytearray(b"\xAA" * 12)
    p.payload_len = len(p.payload)
    return p


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(disp_mod.time, "monotonic", c.monotonic)
    return c


# --------------------------------------------------------------------------- #
# Budget arithmetic (independently computed firmware values)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "factor,duty",
    [(0.0, 1.0), (1.0, 0.5), (3.0, 0.25), (9.0, 0.1)],
)
def test_duty_cycle_from_airtime_factor(factor, duty):
    d, _ = _make(factor=factor, enabled=False)
    assert d._duty_cycle() == pytest.approx(duty)


def test_reset_starts_full(clock):
    # Dispatcher::begin: tx_budget_ms = window * duty_cycle. factor 1.0 -> 50%.
    d, _ = _make(factor=1.0)
    assert d._tx_budget_ms == pytest.approx(DUTY_CYCLE_WINDOW_MS * 0.5)  # 1_800_000


def test_refill_accrues_at_duty_and_caps(clock):
    d, _ = _make(factor=1.0)  # duty 0.5, max 1_800_000
    d._tx_budget_ms = 0.0
    d._tx_budget_last_update = clock.t
    # Advance 1000 s: refill = elapsed_ms * duty = 1_000_000 * 0.5 = 500_000.
    clock.t += 1000.0
    d._refill_tx_budget(clock.t)
    assert d._tx_budget_ms == pytest.approx(500_000.0)
    # Advance far beyond the window: budget caps at window * duty.
    clock.t += 100_000.0
    d._refill_tx_budget(clock.t)
    assert d._tx_budget_ms == pytest.approx(DUTY_CYCLE_WINDOW_MS * 0.5)


def test_debit_uses_actual_airtime(clock):
    d, _ = _make(factor=1.0)
    d._tx_est_airtime_ms = lambda n: 123.0  # controlled per-packet airtime
    d._tx_budget_ms = 1000.0
    d._tx_budget_last_update = clock.t  # no refill (elapsed 0)
    d._debit_tx_budget(_flood_txt())
    assert d._tx_budget_ms == pytest.approx(877.0)  # 1000 - 123


def test_debit_clamps_at_zero_and_sets_pacing(clock):
    d, _ = _make(factor=1.0)
    d._tx_est_airtime_ms = lambda n: 500.0
    d._tx_budget_ms = 200.0  # less than the airtime to spend
    d._tx_budget_last_update = clock.t
    d._debit_tx_budget(_flood_txt())
    assert d._tx_budget_ms == 0.0
    # budget < MIN_TX_BUDGET_RESERVE_MS -> next_tx_time = now + needed/duty.
    needed = MIN_TX_BUDGET_RESERVE_MS - 0.0
    assert d._tx_next_time == pytest.approx(clock.t + (needed / 0.5) / 1000.0)


# --------------------------------------------------------------------------- #
# TX gate: delay-not-drop, boundary
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_gate_no_wait_when_budget_at_reserve(clock, monkeypatch):
    monkeypatch.setattr(disp_mod.asyncio, "sleep", clock.sleep)
    d, _ = _make(factor=1.0)
    d._tx_est_airtime_ms = lambda n: 200.0  # reserve = 200/2 = 100
    d._tx_budget_ms = 100.0  # exactly at reserve -> not below -> no wait
    d._tx_next_time = clock.t
    await d._await_tx_budget(_flood_txt())
    assert clock.slept == []  # returned immediately


@pytest.mark.asyncio
async def test_gate_waits_computed_amount_when_short(clock, monkeypatch):
    monkeypatch.setattr(disp_mod.asyncio, "sleep", clock.sleep)
    d, _ = _make(factor=1.0)  # duty 0.5
    d._tx_est_airtime_ms = lambda n: 200.0  # reserve = 100
    d._tx_budget_ms = 60.0  # 40 ms short of reserve
    d._tx_next_time = clock.t
    await d._await_tx_budget(_flood_txt())
    # needed 40 ms / duty 0.5 = 80 ms = 0.08 s; after that refill reaches reserve.
    assert clock.slept == [pytest.approx(0.08)]
    assert d._tx_budget_ms == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_burst_delayed_not_dropped(clock, monkeypatch):
    monkeypatch.setattr(disp_mod.asyncio, "sleep", clock.sleep)
    d, radio = _make(factor=1.0)
    d._tx_est_airtime_ms = lambda n: 200.0  # reserve 100, spend 200 each
    d._tx_budget_ms = 300.0  # only enough for the first without waiting
    d._tx_next_time = clock.t
    for _ in range(4):
        assert await d.send_packet(_flood_txt(), wait_for_ack=False) is True
    # All four eventually transmit (never dropped); the short budget forced waits.
    assert radio.send_count == 4
    assert len(clock.slept) >= 1
    assert sum(s for s in clock.slept if s > 0) > 0


# --------------------------------------------------------------------------- #
# Repeat-off: send path untouched
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_repeat_off_send_does_not_gate_or_debit(monkeypatch):
    d, radio = _make(enabled=False)
    called = {"await": 0, "debit": 0}

    async def _spy_await(pkt):
        called["await"] += 1

    # Budget entry points are the only send-path callers of time.monotonic;
    # proving they are not entered proves the hot path adds no time syscalls.
    monkeypatch.setattr(d, "_await_tx_budget", _spy_await)
    monkeypatch.setattr(d, "_debit_tx_budget", lambda pkt: called.__setitem__("debit", 1))

    assert await d.send_packet(_flood_txt(), wait_for_ack=False) is True
    assert radio.send_count == 1
    assert called == {"await": 0, "debit": 0}  # hot path untouched


# --------------------------------------------------------------------------- #
# Cancellation safety
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cancellation_during_wait_leaves_bucket_consistent(clock):
    # Real asyncio.sleep here so the wait actually suspends and can be cancelled.
    d, radio = _make(factor=1.0)
    d._tx_est_airtime_ms = lambda n: 200.0  # reserve 100
    d._tx_budget_ms = 0.0  # far short -> long wait
    d._tx_budget_last_update = clock.t
    d._tx_next_time = clock.t

    task = asyncio.create_task(d._await_tx_budget(_flood_txt()))
    await asyncio.sleep(0)  # let it reach the real sleep
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # No transmit happened and the bucket is unchanged/consistent: budget stayed
    # at the last synchronous refill value (0.0, elapsed was 0) and last_update
    # equals the refill time. No partial debit.
    assert radio.send_count == 0
    assert d._tx_budget_ms == 0.0
    assert d._tx_budget_last_update == clock.t
