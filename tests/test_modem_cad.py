from unittest.mock import AsyncMock

import pytest

from openhop_core.hardware.protocol_constants import (
    CMD_CAD_PARAMS_RESP,
    CMD_SET_CAD_PARAMS,
)
from openhop_core.hardware.tcp_radio import TCPLoRaRadio
from openhop_core.hardware.usb_radio import USBLoRaRadio


def _make_radio(radio_cls):
    if radio_cls is TCPLoRaRadio:
        return radio_cls("127.0.0.1")
    return radio_cls("/dev/null")


@pytest.mark.asyncio
@pytest.mark.parametrize("radio_cls", [TCPLoRaRadio, USBLoRaRadio])
async def test_modem_cad_accepts_symbol_count_and_programs_firmware(radio_cls):
    radio = _make_radio(radio_cls)
    radio._send_command = AsyncMock(return_value=b"\x01")
    radio._perform_cad = AsyncMock(return_value=False)

    result = await radio.perform_cad(
        det_peak=23,
        det_min=11,
        timeout=0.5,
        calibration=True,
        cad_symbol_num=8,
    )

    assert result is False
    radio._send_command.assert_awaited_once_with(
        CMD_SET_CAD_PARAMS,
        bytes([0x03, 23, 11, 0x00]),
        expect_cmd=CMD_CAD_PARAMS_RESP,
        timeout=2.0,
    )
    radio._perform_cad.assert_awaited_once_with(0.6)


@pytest.mark.asyncio
@pytest.mark.parametrize("radio_cls", [TCPLoRaRadio, USBLoRaRadio])
async def test_modem_cad_reprograms_firmware_when_only_symbol_count_changes(radio_cls):
    radio = _make_radio(radio_cls)
    radio._send_command = AsyncMock(return_value=b"\x01")
    radio._perform_cad = AsyncMock(return_value=False)

    await radio.perform_cad(det_peak=23, det_min=11, cad_symbol_num=2)
    await radio.perform_cad(det_peak=23, det_min=11, cad_symbol_num=8)

    payloads = [call.args[1] for call in radio._send_command.await_args_list]
    assert payloads == [bytes([0x01, 23, 11, 0x00]), bytes([0x03, 23, 11, 0x00])]


@pytest.mark.parametrize("radio_cls", [TCPLoRaRadio, USBLoRaRadio])
def test_modem_cad_symbol_setter_tracks_valid_symbol_count_before_start(radio_cls):
    radio = _make_radio(radio_cls)

    assert radio.set_custom_cad_symbol_num(16) is True
    assert radio._custom_cad_symbol_num == 16


@pytest.mark.parametrize("radio_cls", [TCPLoRaRadio, USBLoRaRadio])
def test_modem_cad_symbol_setter_rejects_unsupported_count(radio_cls):
    radio = _make_radio(radio_cls)

    with pytest.raises(ValueError, match="cad_symbol_num must be one of"):
        radio.set_custom_cad_symbol_num(3)
