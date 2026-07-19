"""
Tests for KissSerialWrapper (generic KISS TNC over serial).

Decoder and queue behavior is exercised without a real serial port.
"""

from openhop_core.hardware.kiss_serial_wrapper import (
    KISS_CMD_DATA,
    KISS_FEND,
    KISS_FESC,
    KISS_TFEND,
    KISS_TFESC,
    MAX_FRAME_SIZE,
    KissSerialWrapper,
)


def _make_wrapper(**kwargs) -> KissSerialWrapper:
    defaults = {
        "port": "/dev/null",
        "auto_configure": False,
    }
    defaults.update(kwargs)
    return KissSerialWrapper(**defaults)


def _kiss_encode(cmd: int, data: bytes, kiss_port: int = 0) -> bytes:
    frame = bytearray([KISS_FEND, ((kiss_port & 0x0F) << 4) | (cmd & 0x0F)])
    for byte in data:
        if byte == KISS_FEND:
            frame.extend([KISS_FESC, KISS_TFEND])
        elif byte == KISS_FESC:
            frame.extend([KISS_FESC, KISS_TFESC])
        else:
            frame.append(byte)
    frame.append(KISS_FEND)
    return bytes(frame)


def _feed_bytewise(wrapper: KissSerialWrapper, data: bytes) -> None:
    for b in data:
        wrapper._decode_kiss_byte(b)


def _feed_chunks(wrapper: KissSerialWrapper, data: bytes, chunk_sizes) -> None:
    i = 0
    for size in chunk_sizes:
        wrapper._decode_kiss(data[i : i + size])
        i += size
    if i < len(data):
        wrapper._decode_kiss(data[i:])


class TestOversizeFrameResync:
    """Unterminated RX frames must resync at MAX_FRAME_SIZE, not grow unbounded."""

    def test_bytewise_oversize_then_valid(self):
        received = []
        wrapper = _make_wrapper(on_frame_received=received.append)
        runaway = bytes([KISS_FEND, KISS_CMD_DATA]) + bytes(MAX_FRAME_SIZE + 50)
        valid = _kiss_encode(KISS_CMD_DATA, b"\x01\x02")
        _feed_bytewise(wrapper, runaway + valid)

        assert len(received) == 1
        assert received[0] == b"\x01\x02"
        assert wrapper.stats["frame_errors"] >= 1
        assert len(wrapper.rx_frame_buffer) <= MAX_FRAME_SIZE
        assert not wrapper.in_frame or len(wrapper.rx_frame_buffer) <= MAX_FRAME_SIZE

    def test_bulk_oversize_whole_stream(self):
        received = []
        wrapper = _make_wrapper(on_frame_received=received.append)
        runaway = bytes([KISS_FEND, KISS_CMD_DATA]) + bytes(MAX_FRAME_SIZE + 50)
        valid = _kiss_encode(KISS_CMD_DATA, b"\xaa\xbb")
        wrapper._decode_kiss(runaway + valid)

        assert len(received) == 1
        assert received[0] == b"\xaa\xbb"
        assert wrapper.stats["frame_errors"] >= 1
        assert len(wrapper.rx_frame_buffer) <= MAX_FRAME_SIZE

    def test_bulk_oversize_chunked(self):
        received = []
        wrapper = _make_wrapper(on_frame_received=received.append)
        runaway = bytes([KISS_FEND, KISS_CMD_DATA]) + bytes(MAX_FRAME_SIZE + 50)
        valid = _kiss_encode(KISS_CMD_DATA, b"\x03\x04")
        stream = runaway + valid
        _feed_chunks(wrapper, stream, [7, 600, 50])

        assert len(received) == 1
        assert received[0] == b"\x03\x04"
        assert wrapper.stats["frame_errors"] >= 1
        assert len(wrapper.rx_frame_buffer) <= MAX_FRAME_SIZE

    def test_buffer_never_exceeds_max_during_runaway(self):
        wrapper = _make_wrapper()
        wrapper._decode_kiss_byte(KISS_FEND)
        for _ in range(MAX_FRAME_SIZE + 200):
            wrapper._decode_kiss_byte(0x55)
            assert len(wrapper.rx_frame_buffer) <= MAX_FRAME_SIZE

        assert wrapper.stats["frame_errors"] >= 1
        assert not wrapper.in_frame

    def test_invalid_escape_resync_then_valid(self):
        received = []
        wrapper = _make_wrapper(on_frame_received=received.append)
        bad = bytes([KISS_FEND, KISS_CMD_DATA, KISS_FESC, 0x42, 0x99, KISS_FEND])
        valid = _kiss_encode(KISS_CMD_DATA, b"\x07")
        wrapper._decode_kiss(bad + valid)

        assert len(received) == 1
        assert received[0] == b"\x07"
        assert wrapper.stats["frame_errors"] >= 1
