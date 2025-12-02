# The JENOVA Cognitive Architecture - Peer Management
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Peer connection lifecycle and capability management.

This module manages connections to discovered peers, tracks their capabilities,
implements load balancing, and handles failover scenarios.
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

from jenova.network.discovery import PeerInfo


class PeerStatus(Enum):
    """Status of a peer connection."""

    UNKNOWN = "unknown"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"
    FAILED = "failed"


@dataclass
class PeerCapabilities:
    """Capabilities advertised by a peer."""

    instance_id: str
    instance_name: str

    # Resource sharing
    share_llm: bool = False
    share_embeddings: bool = False
    share_memory: bool = False

    # Hardware
    gpu_layers: int = 0
    context_size: int = 4096
    cpu_cores: int = 0
    total_ram_mb: int = 0

    # Current load
    current_load_percent: int = 0
    active_requests: int = 0

    # Version
    version: str = "unknown"
    protocol_version: str = "1.0"

    # Performance metrics
    avg_response_time_ms: float = 0.0
    success_rate_percent: float = 100.0


@dataclass
class PeerConnection:
    """Represents a connection to a peer."""

    peer_info: PeerInfo
    capabilities: Optional[PeerCapabilities] = None
    status: PeerStatus = PeerStatus.UNKNOWN
    last_health_check: float = 0.0
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    max_response_times: int = 100  # Keep last 100 response times


def _peer_info_from_dict(data: Dict) -> PeerInfo:
    """Convert a dictionary to PeerInfo object for backward compatibility."""
    import time
    return PeerInfo(
        instance_id=data.get("id", data.get("instance_id", "")),
        instance_name=data.get("name", data.get("instance_name", "")),
        address=data.get("host", data.get("address", "")),
        port=data.get("port", 50051),
        properties=data,
        last_seen=data.get("last_seen") if isinstance(data.get("last_seen"), (int, float)) else time.time(),
    )


class PeerManager:
    """
    Manages peer connections and capabilities.

    Responsibilities:
    - Track peer status and capabilities
    - Health monitoring
    - Load balancing
    - Failover handling
    - Connection pooling
    """

    def __init__(self, config: dict, file_logger, ui_logger=None):
        """
        Initialize peer manager.

        Args:
            config: JENOVA configuration
            file_logger: Logger for file output
            ui_logger: Optional UI logger
        """
        self.config = config
        self.file_logger = file_logger
        self.ui_logger = ui_logger

        # Peer connections
        self.peers: Dict[str, PeerConnection] = {}
        self.peers_lock = threading.RLock()

        # Configuration
        network_config = config.get("network", {})
        peer_selection = network_config.get("peer_selection", {})
        self.strategy = peer_selection.get("strategy", "load_balanced")
        self.timeout_ms = peer_selection.get("timeout_ms", 5000)

        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.max_consecutive_failures = 3
        self.running = False
        self.health_monitor_thread = None

    def start(self):
        """Start the peer manager."""
        self.file_logger.log_info("Starting peer manager...")
        self.running = True

        # Start health monitoring
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop, daemon=True
        )
        self.health_monitor_thread.start()

        self.file_logger.log_info(f"Peer manager started (strategy: {self.strategy})")

    def stop(self):
        """Stop the peer manager."""
        self.file_logger.log_info("Stopping peer manager...")
        self.running = False

        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            self.health_monitor_thread.join(timeout=5)

        self.file_logger.log_info("Peer manager stopped")

    def add_peer(self, peer_info: Union[PeerInfo, Dict]):
        """
        Add a new peer.

        Args:
            peer_info: Information about the discovered peer (PeerInfo or dict)
        """
        # Support both PeerInfo objects and dicts for backward compatibility
        if isinstance(peer_info, dict):
            peer_info = _peer_info_from_dict(peer_info)
        
        with self.peers_lock:
            if peer_info.instance_id not in self.peers:
                connection = PeerConnection(
                    peer_info=peer_info, status=PeerStatus.CONNECTING
                )
                self.peers[peer_info.instance_id] = connection

                self.file_logger.log_info(
                    f"Added peer: {peer_info.instance_name} ({peer_info.instance_id})"
                )

    def remove_peer(self, peer_id: str):
        """
        Remove a peer.

        Args:
            peer_id: Instance ID of peer to remove
        """
        with self.peers_lock:
            if peer_id in self.peers:
                peer = self.peers[peer_id]
                del self.peers[peer_id]

                self.file_logger.log_info(
                    f"Removed peer: {peer.peer_info.instance_name}"
                )

    def update_peer_capabilities(self, peer_id: str, capabilities: PeerCapabilities):
        """
        Update capabilities for a peer.

        Args:
            peer_id: Instance ID
            capabilities: Updated capabilities
        """
        with self.peers_lock:
            if peer_id in self.peers:
                self.peers[peer_id].capabilities = capabilities
                self.peers[peer_id].status = PeerStatus.CONNECTED
                self.file_logger.log_info(
                    f"Updated capabilities for peer: {capabilities.instance_name}"
                )

    def record_request_result(
        self, peer_id: str, success: bool, response_time_ms: float
    ):
        """
        Record the result of a request to a peer.

        Args:
            peer_id: Instance ID
            success: Whether request succeeded
            response_time_ms: Response time in milliseconds
        """
        with self.peers_lock:
            if peer_id not in self.peers:
                return

            peer = self.peers[peer_id]
            peer.total_requests += 1

            if success:
                peer.successful_requests += 1
                peer.consecutive_failures = 0
            else:
                peer.failed_requests += 1
                peer.consecutive_failures += 1

                # Mark as degraded/failed based on failures
                if peer.consecutive_failures >= self.max_consecutive_failures:
                    peer.status = PeerStatus.FAILED
                    self.file_logger.log_warning(
                        f"Peer marked as failed: {peer.peer_info.instance_name} "
                        f"({peer.consecutive_failures} consecutive failures)"
                    )
                elif peer.consecutive_failures >= 1:
                    peer.status = PeerStatus.DEGRADED

            # Track response times
            peer.response_times.append(response_time_ms)
            if len(peer.response_times) > peer.max_response_times:
                peer.response_times.pop(0)

    def select_peer_for_task(
        self, task_type: str, exclude_peers: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Select the best peer for a task based on strategy.

        Args:
            task_type: Type of task ('llm', 'embedding', 'memory')
            exclude_peers: List of peer IDs to exclude

        Returns:
            Selected peer ID or None if no suitable peer
        """
        with self.peers_lock:
            # Filter available peers
            available_peers = []
            for peer_id, peer in self.peers.items():
                # Skip excluded peers
                if exclude_peers and peer_id in exclude_peers:
                    continue

                # Check status
                if peer.status not in [PeerStatus.CONNECTED, PeerStatus.DEGRADED]:
                    continue

                # Check capabilities
                if not peer.capabilities:
                    continue

                # Check task-specific capability
                if task_type == "llm" and not peer.capabilities.share_llm:
                    continue
                if task_type == "embedding" and not peer.capabilities.share_embeddings:
                    continue
                if task_type == "memory" and not peer.capabilities.share_memory:
                    continue

                available_peers.append(peer_id)

            if not available_peers:
                return None

            # Apply selection strategy
            if self.strategy == "load_balanced":
                return self._select_by_load(available_peers)
            elif self.strategy == "fastest":
                return self._select_by_speed(available_peers)
            elif self.strategy == "local_first":
                # In LAN context, "local_first" means prefer lower latency
                return self._select_by_speed(available_peers)
            else:
                # Default: round-robin
                return available_peers[0]

    def _select_by_load(self, peer_ids: List[str]) -> str:
        """Select peer with lowest current load."""
        best_peer = None
        lowest_load = float("inf")

        for peer_id in peer_ids:
            peer = self.peers[peer_id]
            if not peer.capabilities:
                continue

            load = peer.capabilities.current_load_percent
            if load < lowest_load:
                lowest_load = load
                best_peer = peer_id

        return best_peer or peer_ids[0]

    def _select_by_speed(self, peer_ids: List[str]) -> str:
        """Select peer with best average response time."""
        best_peer = None
        best_time = float("inf")

        for peer_id in peer_ids:
            peer = self.peers[peer_id]

            # Calculate average response time
            if peer.response_times:
                avg_time = sum(peer.response_times) / len(peer.response_times)
            else:
                # Untested peers get infinity - they'll be tried if no tested peers available
                # but won't be prioritized over peers with known good latency
                avg_time = float("inf")

            if avg_time < best_time:
                best_time = avg_time
                best_peer = peer_id

        return best_peer or peer_ids[0]

    def get_peer_connection(self, peer_id: str) -> Optional[PeerConnection]:
        """Get connection info for a peer."""
        with self.peers_lock:
            return self.peers.get(peer_id)

    def get_all_peers(self) -> List[PeerConnection]:
        """Get all peer connections."""
        with self.peers_lock:
            return list(self.peers.values())

    def get_connected_peer_count(self) -> int:
        """Get count of connected peers."""
        with self.peers_lock:
            return sum(
                1 for peer in self.peers.values() if peer.status == PeerStatus.CONNECTED
            )

    def _health_monitor_loop(self):
        """Background thread for health monitoring."""
        while self.running:
            try:
                time.sleep(self.health_check_interval)

                with self.peers_lock:
                    for peer_id, peer in list(self.peers.items()):
                        # Check if peer is stale
                        time_since_update = time.time() - peer.peer_info.last_seen
                        if time_since_update > 90:  # 90 seconds stale threshold
                            self.file_logger.log_warning(
                                f"Peer appears stale: {peer.peer_info.instance_name} "
                                f"(not updated in {time_since_update:.0f}s)"
                            )
                            if peer.status == PeerStatus.CONNECTED:
                                peer.status = PeerStatus.DEGRADED

            except Exception as e:
                self.file_logger.log_error(f"Error in peer health monitor: {e}")
                time.sleep(5)

    def get_status(self) -> Dict:
        """Get peer manager status."""
        with self.peers_lock:
            status_counts = defaultdict(int)
            for peer in self.peers.values():
                status_counts[peer.status.value] += 1

            return {
                "total_peers": len(self.peers),
                "status_breakdown": dict(status_counts),
                "strategy": self.strategy,
                "timeout_ms": self.timeout_ms,
                "peers": [
                    {
                        "instance_id": peer.peer_info.instance_id,
                        "instance_name": peer.peer_info.instance_name,
                        "address": f"{peer.peer_info.address}:{peer.peer_info.port}",
                        "status": peer.status.value,
                        "total_requests": peer.total_requests,
                        "success_rate": (
                            peer.successful_requests / peer.total_requests * 100
                            if peer.total_requests > 0
                            else 0
                        ),
                        "avg_response_time_ms": (
                            sum(peer.response_times) / len(peer.response_times)
                            if peer.response_times
                            else 0
                        ),
                    }
                    for peer in self.peers.values()
                ],
            }

    def get_active_peers(self) -> List[Dict]:
        """
        Get list of active (connected or degraded) peers.
        
        Returns:
            List of peer dictionaries with their details
        """
        with self.peers_lock:
            return [
                {
                    "id": peer.peer_info.instance_id,
                    "instance_id": peer.peer_info.instance_id,
                    "instance_name": peer.peer_info.instance_name,
                    "host": peer.peer_info.address,
                    "port": peer.peer_info.port,
                    "status": peer.status.value,
                    "health": peer.status.value if peer.status in [PeerStatus.CONNECTED] else "degraded" if peer.status == PeerStatus.DEGRADED else "unhealthy",
                    "load": peer.capabilities.current_load_percent if peer.capabilities else 0,
                    "latency_ms": sum(peer.response_times) / len(peer.response_times) if peer.response_times else 0,
                    "capabilities": [
                        cap for cap in ["llm", "embeddings", "memory"]
                        if peer.capabilities and getattr(peer.capabilities, f"share_{cap}", False)
                    ] if peer.capabilities else [],
                    "last_seen": peer.peer_info.last_seen,
                }
                for peer in self.peers.values()
                if peer.status in [PeerStatus.CONNECTED, PeerStatus.DEGRADED]
            ]

    def get_peer(self, peer_id: str) -> Optional[Dict]:
        """
        Get peer details by ID as a dictionary.
        
        Args:
            peer_id: The peer instance ID
            
        Returns:
            Peer details dictionary or None if not found
        """
        with self.peers_lock:
            if peer_id not in self.peers:
                return None
            peer = self.peers[peer_id]
            return {
                "id": peer.peer_info.instance_id,
                "instance_id": peer.peer_info.instance_id,
                "instance_name": peer.peer_info.instance_name,
                "host": peer.peer_info.address,
                "port": peer.peer_info.port,
                "status": peer.status.value,
                "health": peer.peer_info.properties.get("health", "unknown"),
                "load": peer.capabilities.current_load_percent if peer.capabilities else peer.peer_info.properties.get("load", 0),
                "latency_ms": sum(peer.response_times) / len(peer.response_times) if peer.response_times else peer.peer_info.properties.get("latency_ms", 0),
                "capabilities": peer.peer_info.properties.get("capabilities", []),
                "last_seen": peer.peer_info.last_seen,
            }

    def update_peer_health(self, peer_id: str, health_status: str):
        """
        Update the health status of a peer.
        
        Args:
            peer_id: The peer instance ID
            health_status: Health status ('healthy', 'degraded', 'unhealthy')
        """
        with self.peers_lock:
            if peer_id in self.peers:
                peer = self.peers[peer_id]
                # Store health in properties
                peer.peer_info.properties["health"] = health_status
                # Update status based on health
                if health_status == "healthy":
                    peer.status = PeerStatus.CONNECTED
                elif health_status == "degraded":
                    peer.status = PeerStatus.DEGRADED
                elif health_status == "unhealthy":
                    peer.status = PeerStatus.FAILED
                self.file_logger.log_info(
                    f"Updated peer {peer_id} health to: {health_status}"
                )

    def update_peer_load(self, peer_id: str, load_percent: float):
        """
        Update the load percentage of a peer.
        
        Args:
            peer_id: The peer instance ID
            load_percent: Load percentage (0.0 - 1.0)
        """
        with self.peers_lock:
            if peer_id in self.peers:
                peer = self.peers[peer_id]
                # Store load in properties and capabilities
                peer.peer_info.properties["load"] = load_percent
                if peer.capabilities:
                    peer.capabilities.current_load_percent = int(load_percent * 100)
                self.file_logger.log_info(
                    f"Updated peer {peer_id} load to: {load_percent}"
                )

    def select_peer(self, strategy: Optional[str] = None) -> Optional[Dict]:
        """
        Select a peer based on the given strategy.
        
        Args:
            strategy: Selection strategy ('load_balanced', 'fastest', 'round_robin')
                     If None, uses the configured default strategy.
        
        Returns:
            Selected peer details dictionary or None if no peers available
        """
        with self.peers_lock:
            active_peers = [
                peer for peer in self.peers.values()
                if peer.status in [PeerStatus.CONNECTED, PeerStatus.DEGRADED]
            ]
            
            if not active_peers:
                return None
            
            effective_strategy = strategy or self.strategy
            
            if effective_strategy == "load_balanced":
                # Select peer with lowest load
                selected = min(
                    active_peers,
                    key=lambda p: p.capabilities.current_load_percent if p.capabilities else float('inf')
                )
            elif effective_strategy == "fastest":
                # Select peer with lowest latency
                selected = min(
                    active_peers,
                    key=lambda p: sum(p.response_times) / len(p.response_times) if p.response_times else float('inf')
                )
            else:
                # Default: round-robin (just take the first one)
                selected = active_peers[0]
            
            return {
                "id": selected.peer_info.instance_id,
                "host": selected.peer_info.address,
                "port": selected.peer_info.port,
            }

    def get_peers_with_capability(self, capability: str) -> List[Dict]:
        """
        Get peers that have a specific capability.
        
        Args:
            capability: Capability to filter by ('llm', 'embeddings', 'memory')
            
        Returns:
            List of peer dictionaries with the specified capability
        """
        with self.peers_lock:
            result = []
            for peer in self.peers.values():
                # Check capabilities from the PeerCapabilities dataclass
                has_capability = False
                if peer.capabilities:
                    if capability == "llm" and peer.capabilities.share_llm:
                        has_capability = True
                    elif capability == "embeddings" and peer.capabilities.share_embeddings:
                        has_capability = True
                    elif capability == "memory" and peer.capabilities.share_memory:
                        has_capability = True
                
                # Also check properties for backwards compatibility
                if not has_capability and "capabilities" in peer.peer_info.properties:
                    if capability in peer.peer_info.properties.get("capabilities", []):
                        has_capability = True
                
                if has_capability:
                    result.append({
                        "id": peer.peer_info.instance_id,
                        "instance_id": peer.peer_info.instance_id,
                        "instance_name": peer.peer_info.instance_name,
                        "host": peer.peer_info.address,
                        "port": peer.peer_info.port,
                    })
            return result
