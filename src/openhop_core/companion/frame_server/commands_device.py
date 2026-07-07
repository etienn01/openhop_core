"""Device command handlers: app start/device query, time, radio params,
stats, flood scope, custom vars, autoadd, keys, tuning."""

import asyncio
import inspect
import logging
import struct

from ...protocol import CryptoUtils
from ..constants import (
    ADV_TYPE_CHAT,
    ERR_CODE_BAD_STATE,
    ERR_CODE_ILLEGAL_ARG,
    FIRMWARE_VER_CODE,
    RESP_CODE_ALLOWED_REPEAT_FREQ,
    RESP_CODE_AUTOADD_CONFIG,
    RESP_CODE_BATT_AND_STORAGE,
    RESP_CODE_CURR_TIME,
    RESP_CODE_CUSTOM_VARS,
    RESP_CODE_DEFAULT_FLOOD_SCOPE,
    RESP_CODE_DEVICE_INFO,
    RESP_CODE_PRIVATE_KEY,
    RESP_CODE_SELF_INFO,
    RESP_CODE_STATS,
    STATS_TYPE_CORE,
    STATS_TYPE_PACKETS,
    STATS_TYPE_RADIO,
)

logger = logging.getLogger("CompanionFrameServer")


class _DeviceCommandsMixin:
    """Device and configuration _cmd_* handlers of :class:`CompanionFrameServer`."""

    async def _cmd_app_start(self, data: bytes) -> None:
        if len(data) >= 1:
            self._app_target_ver = data[0]
        prefs = self.bridge.get_self_info()
        pubkey = self.bridge.get_public_key()
        name = prefs.node_name.encode("utf-8", errors="replace")
        lat = int(getattr(prefs, "latitude", 0) * 1e6)
        lon = int(getattr(prefs, "longitude", 0) * 1e6)
        frame = (
            bytes([RESP_CODE_SELF_INFO, ADV_TYPE_CHAT, prefs.tx_power_dbm, 22])
            + pubkey
            + struct.pack("<ii", lat, lon)
            + bytes(
                [
                    getattr(prefs, "multi_acks", 0),
                    getattr(prefs, "advert_loc_policy", 0),
                ]
            )
            + bytes(
                [
                    getattr(prefs, "telemetry_mode_base", 0)
                    | (getattr(prefs, "telemetry_mode_location", 0) << 2)
                ]
            )
            + bytes([getattr(prefs, "manual_add_contacts", 0)])
            + struct.pack(
                "<II",
                prefs.frequency_hz // 1000,
                prefs.bandwidth_hz,
            )
            + bytes([prefs.spreading_factor, prefs.coding_rate])
            + name
        )
        self._write_frame(frame)

    async def _cmd_device_query(self, data: bytes) -> None:
        # Layout must match MeshCore companion_radio MyMesh.cpp handleCmdFrame() CMD_DEVICE_QUEURY:
        # [0]=RESP_CODE_DEVICE_INFO, [1]=FIRMWARE_VER_CODE, [2]=MAX_CONTACTS/2,
        # [3]=MAX_GROUP_CHANNELS, [4..7]=ble_pin, [8..19]=build_date(12), [20..59]=manufacturer(40),
        # [60..79]=version(20), [80]=client_repeat, [81]=path_hash_mode (v10+).
        if len(data) >= 1:
            self._app_target_ver = data[0]
        firmware_ver = FIRMWARE_VER_CODE
        max_contacts = getattr(getattr(self.bridge, "contacts", None), "max_contacts", 1000)
        max_channels_val = getattr(getattr(self.bridge, "channels", None), "max_channels", 40)
        max_contacts_div_2 = min(max_contacts // 2, 255)
        max_channels = min(max_channels_val, 255)
        ble_pin = 0
        try:
            prefs = self.bridge.get_self_info()
            client_repeat = getattr(prefs, "client_repeat", 0) & 0xFF
            path_hash_mode = getattr(prefs, "path_hash_mode", 0) & 0xFF
        except Exception:
            client_repeat = 0
            path_hash_mode = 0
        frame = (
            bytes(
                [
                    RESP_CODE_DEVICE_INFO,
                    firmware_ver,
                    max_contacts_div_2,
                    max_channels,
                ]
            )
            + struct.pack("<I", ble_pin)
            + self._build_date_bytes
            + self._model_bytes
            + self._version_bytes
            + bytes([client_repeat & 0xFF, path_hash_mode & 0xFF])
        )
        version_str = self._version_bytes.split(b"\x00")[0].decode("utf-8", errors="replace")
        logger.info(
            "Companion device info sent: FIRMWARE_VER_CODE=%s (byte at index 1), "
            "version string=%r, frame_len=%s",
            firmware_ver,
            version_str,
            len(frame),
        )
        self._write_frame(frame)

    async def _cmd_send_self_advert(self, data: bytes) -> None:
        flood = len(data) >= 1 and data[0] == 1
        ok = await self.bridge.advertise(flood=flood)
        self._write_ok() if ok else self._write_err(ERR_CODE_BAD_STATE)

    async def _cmd_set_advert_name(self, data: bytes) -> None:
        name = data.decode("utf-8", errors="replace").rstrip("\x00")
        self.bridge.set_advert_name(name)
        self._write_ok()

    async def _cmd_set_advert_latlon(self, data: bytes) -> None:
        if len(data) < 8:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        lat, lon = struct.unpack_from("<ii", data, 0)
        self.bridge.set_advert_latlon(lat / 1e6, lon / 1e6)
        self._write_ok()

    async def _cmd_get_batt_and_storage(self, data: bytes) -> None:
        millivolts, used_kb, total_kb = self._get_batt_and_storage()
        frame = (
            bytes([RESP_CODE_BATT_AND_STORAGE])
            + struct.pack("<H", millivolts)
            + struct.pack("<II", used_kb, total_kb)
        )
        self._write_frame(frame)

    async def _cmd_get_stats(self, data: bytes) -> None:
        stats_type = data[0] if len(data) >= 1 else STATS_TYPE_PACKETS
        if stats_type not in (
            STATS_TYPE_CORE,
            STATS_TYPE_RADIO,
            STATS_TYPE_PACKETS,
        ):
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        if self.stats_getter:
            if inspect.iscoroutinefunction(self.stats_getter):
                stats = await self.stats_getter(stats_type)
            else:
                stats = await asyncio.to_thread(self.stats_getter, stats_type)
        else:
            stats = None
        stats = stats or self.bridge.get_stats(stats_type)
        frame = bytes([RESP_CODE_STATS, stats_type])
        if stats_type == STATS_TYPE_CORE:
            battery_mv = int(stats.get("battery_mv", 0))
            uptime_secs = int(stats.get("uptime_secs", 0))
            errors = int(stats.get("errors", 0))
            queue_len = min(255, max(0, int(stats.get("queue_len", 0))))
            frame += struct.pack("<H I H B", battery_mv, uptime_secs, errors, queue_len)
        elif stats_type == STATS_TYPE_RADIO:
            noise_floor = int(stats.get("noise_floor", 0))
            last_rssi = max(-128, min(127, int(stats.get("last_rssi", 0))))
            last_snr_scaled = max(
                -128,
                min(
                    127,
                    int(round((stats.get("last_snr") or 0) * 4)),
                ),
            )
            tx_air_secs = int(stats.get("tx_air_secs", 0))
            rx_air_secs = int(stats.get("rx_air_secs", 0))
            frame += struct.pack(
                "<h b b I I",
                noise_floor,
                last_rssi,
                last_snr_scaled,
                tx_air_secs,
                rx_air_secs,
            )
        else:
            recv = int(stats.get("recv", 0))
            sent = int(stats.get("sent", 0))
            flood_tx = int(stats.get("flood_tx", 0))
            direct_tx = int(stats.get("direct_tx", 0))
            flood_rx = int(stats.get("flood_rx", 0))
            direct_rx = int(stats.get("direct_rx", 0))
            recv_errors = int(stats.get("recv_errors", 0))
            frame += struct.pack(
                "<I I I I I I I",
                recv,
                sent,
                flood_tx,
                direct_tx,
                flood_rx,
                direct_rx,
                recv_errors,
            )
        self._write_frame(frame)

    async def _cmd_set_flood_scope(self, data: bytes) -> None:
        """Delegate flood scope to the bridge (CMD_SET_FLOOD_SCOPE_KEY).

        Wire format after the cmd byte is stripped: [mode (1)] [key (16)].
        The firmware (MyMesh.cpp:1909) treats data[0] as a mode selector:
          * mode 0: set the scope override key (data[1:17]) when present, else
            reset the override; cancels any pending explicit-unscoped request.
          * mode 1 (FIRMWARE_VER_CODE 12+, PR #2492): force the next flood to be
            unscoped, ignoring the configured default scope.
        Older apps always sent mode 0, so this is backward compatible.
        """
        mode = data[0] if len(data) >= 1 else 0
        if mode == 1:
            self.bridge.set_flood_unscoped()
        elif len(data) >= 17:
            self.bridge.set_flood_scope(data[1:17])
        else:
            self.bridge.set_flood_scope(None)
        self._write_ok()

    async def _cmd_set_default_flood_scope(self, data: bytes) -> None:
        """Handle CMD_SET_DEFAULT_FLOOD_SCOPE (63)."""
        if len(data) < 31 + 16:
            self.bridge.set_default_flood_scope(None, None)
            self._write_ok()
            return
        name_raw = data[:31]
        key = data[31 : 31 + 16]
        scope_name = name_raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()
        if not scope_name or len(scope_name) >= 31:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        if not self.bridge.set_default_flood_scope(scope_name, key):
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        self._write_ok()

    async def _cmd_get_default_flood_scope(self, data: bytes) -> None:
        """Handle CMD_GET_DEFAULT_FLOOD_SCOPE (64)."""
        scope = self.bridge.get_default_flood_scope()
        if scope is None:
            self._write_frame(bytes([RESP_CODE_DEFAULT_FLOOD_SCOPE]))
            return
        name, key = scope
        name_field = name.encode("utf-8", errors="replace")[:30].ljust(31, b"\x00")
        key_field = bytes(key[:16]).ljust(16, b"\x00")
        self._write_frame(bytes([RESP_CODE_DEFAULT_FLOOD_SCOPE]) + name_field + key_field)

    # -------------------------------------------------------------------------
    # Time, radio, tuning, share/export, logout, custom vars, autoadd
    # -------------------------------------------------------------------------

    async def _cmd_get_device_time(self, data: bytes) -> None:
        now = self.bridge.get_time()
        self._write_frame(bytes([RESP_CODE_CURR_TIME]) + struct.pack("<I", now))

    async def _cmd_set_device_time(self, data: bytes) -> None:
        if len(data) < 4:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        secs = struct.unpack("<I", data[:4])[0]
        if self.bridge.set_time(secs):
            self._write_ok()
        else:
            self._write_err(ERR_CODE_ILLEGAL_ARG)

    async def _cmd_set_radio_params(self, data: bytes) -> None:
        if len(data) < 10:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        # Frequency in kHz (match firmware self-info; client sends same encoding)
        freq_khz = struct.unpack_from("<I", data, 0)[0]
        bw = struct.unpack_from("<I", data, 4)[0]
        sf = data[8]
        cr = data[9]
        if not (100_000 <= freq_khz <= 2_500_000):
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        if not (7000 <= bw <= 500000):
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        if not (5 <= sf <= 12) or not (5 <= cr <= 8):
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        self.bridge.set_radio_params(freq_khz * 1000, bw, sf, cr)
        self._write_ok()

    async def _cmd_set_tx_power(self, data: bytes) -> None:
        if len(data) < 1:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        power = struct.unpack_from("<b", data, 0)[0]
        if power < -9 or power >= 30:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        self.bridge.set_tx_power(power)
        self._write_ok()

    async def _cmd_export_private_key(self, data: bytes) -> None:
        """Export private/signing key as 64-byte MeshCore format (RESP_CODE_PRIVATE_KEY + 64 bytes).

        For PyNaCl 32-byte seeds we expand to MeshCore 64-byte format (SHA-512 + clamp) so
        the client's ed25519_derive_pub yields the same public key and signing works.
        """
        identity = self.bridge._identity
        key_bytes = identity.get_signing_key_bytes()
        if len(key_bytes) == 32:
            key_bytes = CryptoUtils.ed25519_expand_seed_to_meshcore_64(key_bytes)
        elif len(key_bytes) < 64:
            key_bytes = key_bytes.ljust(64, b"\x00")
        else:
            key_bytes = key_bytes[:64]
        self._write_frame(bytes([RESP_CODE_PRIVATE_KEY]) + key_bytes)

    async def _cmd_import_private_key(self, data: bytes) -> None:
        """Stub/no-op: private key is set from config; dynamic import may be supported later."""
        self._write_ok()

    async def _cmd_set_tuning_params(self, data: bytes) -> None:
        if len(data) < 8:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        rx_ms = struct.unpack_from("<I", data, 0)[0]
        af_ms = struct.unpack_from("<I", data, 4)[0]
        self.bridge.set_tuning_params(rx_ms / 1000.0, af_ms / 1000.0)
        self._write_ok()

    async def _cmd_get_custom_vars(self, data: bytes) -> None:
        custom_vars = self.bridge.get_custom_vars()
        parts = [f"{k}:{v}" for k, v in custom_vars.items()]
        csv = ",".join(parts)[:140]
        self._write_frame(bytes([RESP_CODE_CUSTOM_VARS]) + csv.encode("utf-8", errors="replace"))

    async def _cmd_set_custom_var(self, data: bytes) -> None:
        if len(data) < 3:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        text = data.split(b"\x00")[0].decode("utf-8", errors="replace")
        sep = text.find(":")
        if sep < 1:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        name = text[:sep]
        value = text[sep + 1 :]
        ok = self.bridge.set_custom_var(name, value)
        self._write_ok() if ok else self._write_err(ERR_CODE_ILLEGAL_ARG)

    async def _cmd_set_autoadd_config(self, data: bytes) -> None:
        if len(data) < 1:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        self.bridge.set_autoadd_config(data[0])
        self._write_ok()

    async def _cmd_get_autoadd_config(self, data: bytes) -> None:
        config = self.bridge.get_autoadd_config()
        self._write_frame(bytes([RESP_CODE_AUTOADD_CONFIG, config & 0xFF]))

    async def _cmd_set_other_params(self, data: bytes) -> None:
        """Handle CMD_SET_OTHER_PARAMS (0x26). Mirrors MyMesh.cpp:1290-1305."""
        if len(data) < 1:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        manual_add = data[0]
        telemetry_modes = data[1] if len(data) >= 2 else 0
        advert_loc_policy = data[2] if len(data) >= 3 else 0
        multi_acks = data[3] if len(data) >= 4 else 0
        self.bridge.set_other_params(manual_add, telemetry_modes, advert_loc_policy, multi_acks)
        self._write_ok()

    async def _cmd_set_path_hash_mode(self, data: bytes) -> None:
        """Handle CMD_SET_PATH_HASH_MODE (61). Format: [subtype(0), mode(0-2)].

        Mirrors MyMesh.cpp:1320-1327.  Subtype byte must be 0; mode values
        0, 1, 2 select 1-byte, 2-byte, 3-byte path hashes respectively.
        """
        if len(data) < 2 or data[0] != 0:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        mode = data[1]
        if mode >= 3:
            self._write_err(ERR_CODE_ILLEGAL_ARG)
            return
        self.bridge.set_path_hash_mode(mode)
        self._write_ok()

    async def _cmd_get_allowed_repeat_freq(self, data: bytes) -> None:
        """Handle CMD_GET_ALLOWED_REPEAT_FREQ (60).

        Firmware (MyMesh.cpp:1958) replies with RESP_ALLOWED_REPEAT_FREQ followed
        by zero or more (lower_freq, upper_freq) little-endian u32 pairs. The
        virtual companion does not model regional repeat-frequency restrictions,
        so it advertises an empty range list (response code with no pairs).
        """
        self._write_frame(bytes([RESP_CODE_ALLOWED_REPEAT_FREQ]))
