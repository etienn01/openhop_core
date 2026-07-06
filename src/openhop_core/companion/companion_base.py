"""
CompanionBase - Shared logic for CompanionRadio and CompanionBridge.

Provides stores, event handling, contact management, device configuration,
and push callbacks. Subclasses implement TX via MeshNode or packet_injector.
The implementation is split across the base_* mixin modules.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional

from ..node.events import EventService, EventSubscriber
from ..protocol import LocalIdentity, Packet
from .base_callbacks import _CallbackMixin
from .base_config import _DeviceConfigMixin
from .base_contacts import _ContactChannelMixin
from .base_events import _RxEventsMixin
from .base_send import _SendOpsMixin

# Re-exported for backwards compatibility (tests and callers import these here).
from .base_support import PUSH_CALLBACK_KEYS  # noqa: F401
from .base_support import ResponseWaiter  # noqa: F401
from .base_support import adv_type_to_flags  # noqa: F401
from .base_support import _SeenCache
from .channel_store import ChannelStore
from .constants import (
    ADV_TYPE_CHAT,
    DEFAULT_MAX_CHANNELS,
    DEFAULT_MAX_CONTACTS,
    DEFAULT_OFFLINE_QUEUE_SIZE,
)
from .contact_store import ContactStore
from .message_queue import MessageQueue
from .models import Contact, NodePrefs
from .path_cache import PathCache
from .stats_collector import StatsCollector

logger = logging.getLogger("CompanionBase")


class _CompanionEventSubscriber(EventSubscriber):
    """Bridges event service to companion push callbacks."""

    def __init__(self, companion: CompanionBase) -> None:
        self._companion = companion

    async def handle_event(self, event_type: str, data: dict) -> None:
        await self._companion._handle_mesh_event(event_type, data)


class CompanionBase(
    _ContactChannelMixin,
    _DeviceConfigMixin,
    _CallbackMixin,
    _SendOpsMixin,
    _RxEventsMixin,
    ABC,
):
    """Abstract base class for companion implementations.

    Provides shared stores, event handling, contact management, device config,
    and push callbacks. Subclasses implement TX (via node or packet_injector).
    """

    def _init_companion_stores(
        self,
        identity: LocalIdentity,
        node_name: str = "pyMC",
        adv_type: int = ADV_TYPE_CHAT,
        max_contacts: int = DEFAULT_MAX_CONTACTS,
        max_channels: int = DEFAULT_MAX_CHANNELS,
        offline_queue_size: int = DEFAULT_OFFLINE_QUEUE_SIZE,
        radio_config: Optional[dict] = None,
        initial_contacts: Optional[Iterable[Contact]] = None,
    ) -> None:
        """Initialize shared stores, prefs, event service, and push callbacks."""
        self._identity = identity
        self._radio_config = radio_config or {}
        self._running = False

        self.contacts = ContactStore(max_contacts)
        self.channels = ChannelStore(max_channels)
        self.message_queue = MessageQueue(offline_queue_size)
        self.path_cache = PathCache()
        self.stats = StatsCollector()

        self.prefs = NodePrefs(
            node_name=node_name,
            adv_type=adv_type,
            tx_power_dbm=self._radio_config.get("power", self._radio_config.get("tx_power", 20)),
            frequency_hz=self._radio_config.get("frequency", 915000000),
            bandwidth_hz=self._radio_config.get("bandwidth", 250000),
            spreading_factor=self._radio_config.get("spreading_factor", 10),
            coding_rate=self._radio_config.get("coding_rate", 5),
        )

        self._custom_vars: dict[str, str] = {}
        self._sign_buffer: Optional[bytearray] = None
        self._flood_transport_key: Optional[bytes] = None
        # One-shot "force unscoped flood" flag (FW PR #2492 / FIRMWARE_VER_CODE 12+):
        # when set, the next flood ignores the default scope and floods unscoped.
        self._flood_unscoped: bool = False
        self._time_offset: float = 0.0

        self._event_service = EventService()
        self._event_subscriber = _CompanionEventSubscriber(self)
        self._event_service.subscribe_all(self._event_subscriber)

        self._push_callbacks: dict[str, list[Callable]] = {k: [] for k in PUSH_CALLBACK_KEYS}

        # Pending binary requests by tag (hex) for matching responses
        self._pending_binary_requests: dict[str, dict] = {}
        # Pending path discovery tags for matching responses
        self._pending_discovery_tags: set[int] = set()
        # Pending ACK CRCs for send_confirmed (Bridge and Radio)
        self._pending_ack_crcs: set[int] = set()
        # Fire-and-forget tasks kept alive until done (see _spawn_background_task)
        self._background_tasks: set[asyncio.Task] = set()

        # Per-payload-type dedup caches keyed by packet hash, matching Mesh.cpp
        # (!_tables->hasSeen(pkt)): the companion queues one frame per logical
        # message and reconnects don't re-queue the same packet.
        self._seen_grp_txt = _SeenCache()
        self._seen_txt = _SeenCache()
        self._seen_grp_data = _SeenCache()

        # Allow subclasses to restore persisted preferences on startup.
        self._load_prefs()

        # Optional bulk load of contacts (e.g. from persistence on boot).
        if initial_contacts is not None:
            self.contacts.load_from(initial_contacts)

    # -------------------------------------------------------------------------
    # Preference Persistence Hooks
    # -------------------------------------------------------------------------

    def _save_prefs(self) -> None:
        """Hook: persist the current :attr:`prefs` to stable storage.

        The default implementation is a no-op — preferences live only in
        memory.  Subclasses that need persistence (e.g. backed by SQLite or
        a JSON file) should override this method.

        Called automatically after any preference-mutating method
        (``set_radio_params``, ``set_tx_power``, ``set_tuning_params``,
        ``set_autoadd_config``, ``set_other_params``,
        ``set_advert_name``, ``set_advert_latlon``).
        """

    def _load_prefs(self) -> None:
        """Hook: restore :attr:`prefs` from stable storage on startup.

        The default implementation is a no-op.  Subclasses should override
        to populate :attr:`self.prefs` fields from their persistence layer.

        Called once at the end of :meth:`_init_companion_stores`.
        """

    # -------------------------------------------------------------------------
    # Abstract methods (subclasses must implement)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def _send_packet(self, pkt: Packet, wait_for_ack: bool = False) -> bool:
        """Send a packet via the subclass transport (radio or packet_injector)."""

    @abstractmethod
    async def start(self) -> None:
        """Start the companion."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the companion."""

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Return whether the companion is currently running."""

    @abstractmethod
    def import_private_key(self, key: bytes) -> bool:
        """Import a private key and rebuild the identity."""

    def _get_protocol_response_handler(self) -> Any:
        """Return the protocol response handler, or ``None``.

        Subclasses that support request/response methods (telemetry, status,
        binary request, etc.) must override this to return their handler.
        """
        return None

    def _get_login_response_handler(self) -> Any:
        """Return the login response handler, or ``None``."""
        return None

    def _get_text_handler(self) -> Any:
        """Return the text message handler, or ``None``."""
        return None

    def _apply_multi_acks_pref(self) -> None:
        """Push the current ``multi_acks`` pref into the text handler (best-effort)."""
        th = self._get_text_handler()
        if th is not None and hasattr(th, "set_multi_acks"):
            th.set_multi_acks(getattr(self.prefs, "multi_acks", 0))
