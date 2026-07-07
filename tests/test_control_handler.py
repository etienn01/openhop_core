import struct

import pytest
from openhop_core.node.handlers.control import ControlHandler
from openhop_core.protocol import Packet
from openhop_core.protocol.packet_utils import PathUtils


@pytest.mark.asyncio
async def test_control_handler_accepts_encoded_zero_hop_discovery_request():
    logs = []
    handler = ControlHandler(log_fn=lambda msg: logs.append(msg))

    pkt = Packet()
    pkt.path_len = PathUtils.encode_path_len(2, 0)  # encoded zero-hop (0x40)
    pkt.payload = bytearray(bytes([0x80, 0x04]) + struct.pack("<I", 0x12345678))
    pkt.payload_len = len(pkt.payload)

    result = await handler(pkt)

    assert result is not None
    assert result["tag"] == 0x12345678
    assert result["filter"] == 0x04
    assert all("Non-zero path length" not in msg for msg in logs)


@pytest.mark.asyncio
async def test_control_handler_rejects_nonzero_hop_discovery_request():
    logs = []
    handler = ControlHandler(log_fn=lambda msg: logs.append(msg))

    pkt = Packet()
    pkt.path_len = PathUtils.encode_path_len(2, 1)  # one hop encoded
    pkt.payload = bytearray(bytes([0x80, 0x04]) + struct.pack("<I", 0x12345678))
    pkt.payload_len = len(pkt.payload)

    result = await handler(pkt)

    assert result is None
    assert any("Non-zero path length" in msg for msg in logs)
