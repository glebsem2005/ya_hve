"""Mesh routing for LoRa network — flooding with deduplication.

Each mic node can relay packets from other nodes, extending range
up to 3 hops x ~10 km = 30 km.  No NB-IoT/4G — forest has no
cellular coverage.

Gateway deduplicates packets by UUID before forwarding to cloud.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_HOPS = 3
DEDUP_TTL_SECONDS = 300  # 5 minutes


@dataclass
class MeshPacket:
    """A packet that can be relayed through the mesh network."""

    packet_id: str  # UUID for deduplication
    source_node: str  # mic_uid of the originator
    hop_count: int
    max_hops: int = MAX_HOPS
    route: list[str] = field(default_factory=list)  # node UIDs traversed
    payload: dict = field(default_factory=dict)

    @classmethod
    def create(cls, source_node: str, payload: dict) -> MeshPacket:
        """Create a new mesh packet from an originating node."""
        return cls(
            packet_id=str(uuid.uuid4()),
            source_node=source_node,
            hop_count=0,
            route=[source_node],
            payload=payload,
        )


class MeshRouter:
    """Flooding-based mesh router with deduplication.

    Used by the gateway to avoid processing the same packet twice
    when it arrives via multiple relay paths.
    """

    def __init__(self) -> None:
        # packet_id -> timestamp of first seen
        self._seen: dict[str, float] = {}

    def _cleanup_seen(self) -> None:
        """Remove expired entries from dedup cache."""
        now = time.monotonic()
        expired = [
            pid for pid, ts in self._seen.items() if now - ts > DEDUP_TTL_SECONDS
        ]
        for pid in expired:
            del self._seen[pid]

    def should_relay(self, packet: MeshPacket) -> bool:
        """Check if this packet should be relayed further.

        Returns False if:
        - Already seen (duplicate)
        - Hop limit reached
        """
        self._cleanup_seen()

        if packet.packet_id in self._seen:
            logger.debug(
                "Mesh: dropping duplicate packet %s from %s",
                packet.packet_id[:8],
                packet.source_node,
            )
            return False

        if packet.hop_count >= packet.max_hops:
            logger.debug(
                "Mesh: packet %s reached max hops (%d)",
                packet.packet_id[:8],
                packet.max_hops,
            )
            return False

        return True

    def process_packet(self, packet: MeshPacket) -> dict | None:
        """Process an incoming mesh packet.

        Returns the payload if this is a new packet (not a duplicate),
        or None if the packet should be dropped.
        """
        if not self.should_relay(packet):
            return None

        # Mark as seen
        self._seen[packet.packet_id] = time.monotonic()

        logger.info(
            "Mesh: accepted packet %s from %s (hop %d/%d, route: %s)",
            packet.packet_id[:8],
            packet.source_node,
            packet.hop_count,
            packet.max_hops,
            " -> ".join(packet.route),
        )

        return packet.payload

    def wrap_for_relay(self, packet: MeshPacket, relay_node: str) -> MeshPacket | None:
        """Prepare a packet for relay by incrementing hop count.

        Returns None if the packet should not be relayed.
        """
        if packet.hop_count >= packet.max_hops:
            return None

        return MeshPacket(
            packet_id=packet.packet_id,
            source_node=packet.source_node,
            hop_count=packet.hop_count + 1,
            max_hops=packet.max_hops,
            route=[*packet.route, relay_node],
            payload=packet.payload,
        )
