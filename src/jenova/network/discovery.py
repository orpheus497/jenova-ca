# The JENOVA Cognitive Architecture - Network Discovery
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
mDNS/Zeroconf-based peer discovery for distributed JENOVA instances.

This module enables automatic discovery of JENOVA instances on the local network
without requiring manual configuration. Instances advertise their capabilities
and can discover each other dynamically.
"""

import socket
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Callable

from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf, ServiceListener


class PeerInfo:
    """Information about a discovered JENOVA peer."""

    def __init__(
        self,
        instance_id: str,
        instance_name: str,
        address: str,
        port: int,
        properties: Dict[str, Any],
        last_seen: float,
    ):
        self.instance_id = instance_id
        self.instance_name = instance_name
        self.address = address
        self.port = port
        self.properties = properties
        self.last_seen = last_seen

    def __repr__(self):
        return f"PeerInfo({self.instance_name}@{self.address}:{self.port})"

    def to_dict(self):
        """Convert to dictionary representation."""
        return {
            "instance_id": self.instance_id,
            "instance_name": self.instance_name,
            "address": self.address,
            "port": self.port,
            "properties": self.properties,
            "last_seen": self.last_seen,
        }


class JenovaServiceListener(ServiceListener):
    """Listener for JENOVA service discovery events."""

    def __init__(self, discovery_service):
        self.discovery_service = discovery_service

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a new service is discovered."""
        info = zc.get_service_info(type_, name)
        if info:
            self.discovery_service._on_service_added(info)

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a service is removed."""
        self.discovery_service._on_service_removed(name)

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a service is updated."""
        info = zc.get_service_info(type_, name)
        if info:
            self.discovery_service._on_service_updated(info)


class JenovaDiscoveryService:
    """
    mDNS/Zeroconf-based service discovery for JENOVA instances.

    This service:
    1. Advertises the local instance on the LAN
    2. Discovers other JENOVA instances
    3. Maintains a registry of available peers
    4. Monitors peer health via periodic updates
    """

    SERVICE_TYPE = "_jenova._tcp.local."

    def __init__(
        self,
        config: Dict,
        file_logger,
        ui_logger=None,
        instance_name: Optional[str] = None,
        port: int = 50051,
    ):
        """
        Initialize discovery service.

        Args:
            config: JENOVA configuration dictionary
            file_logger: Logger for file output
            ui_logger: Optional logger for UI output
            instance_name: Unique name for this instance (auto-generated if None)
            port: gRPC port for this instance
        """
        self.config = config
        self.file_logger = file_logger
        self.ui_logger = ui_logger
        self.port = port

        # Generate unique instance ID
        self.instance_id = str(uuid.uuid4())
        self.instance_name = instance_name or f"jenova-{socket.gethostname()}"

        # Zeroconf instance
        self.zeroconf = None
        self.service_info = None
        self.service_browser = None

        # Peer registry
        self.peers: Dict[str, PeerInfo] = {}
        self.peers_lock = threading.RLock()

        # Callbacks for peer events
        self.on_peer_added_callbacks: List[Callable] = []
        self.on_peer_removed_callbacks: List[Callable] = []
        self.on_peer_updated_callbacks: List[Callable] = []

        # Background threads
        self.running = False
        self.health_monitor_thread = None

        # Configuration
        network_config = config.get("network", {})
        discovery_config = network_config.get("discovery", {})
        self.ttl = discovery_config.get("ttl", 60)
        self.health_check_interval = 30  # seconds

    def start(self):
        """Start the discovery service."""
        try:
            self.file_logger.log_info(
                f"Starting discovery service: {self.instance_name} ({self.instance_id})"
            )

            # Initialize Zeroconf
            self.zeroconf = Zeroconf()

            # Register this instance
            self._register_service()

            # Start browsing for peers
            self.service_browser = ServiceBrowser(
                self.zeroconf, self.SERVICE_TYPE, JenovaServiceListener(self)
            )

            # Start health monitoring
            self.running = True
            self.health_monitor_thread = threading.Thread(
                target=self._health_monitor_loop, daemon=True
            )
            self.health_monitor_thread.start()

            if self.ui_logger:
                self.ui_logger.success(
                    f"Discovery service started: {self.instance_name}"
                )

            self.file_logger.log_info(
                f"Discovery service started successfully on port {self.port}"
            )

        except Exception as e:
            self.file_logger.log_error(f"Failed to start discovery service: {e}")
            raise

    def stop(self):
        """Stop the discovery service."""
        try:
            self.file_logger.log_info("Stopping discovery service...")

            self.running = False

            # Unregister service
            if self.service_info and self.zeroconf:
                self.zeroconf.unregister_service(self.service_info)

            # Stop browser
            if self.service_browser:
                self.service_browser.cancel()

            # Close Zeroconf
            if self.zeroconf:
                self.zeroconf.close()

            # Wait for health monitor to stop
            if self.health_monitor_thread and self.health_monitor_thread.is_alive():
                self.health_monitor_thread.join(timeout=5)

            self.file_logger.log_info("Discovery service stopped")

        except Exception as e:
            self.file_logger.log_error(f"Error stopping discovery service: {e}")

    def _register_service(self):
        """Register this instance as a Zeroconf service."""
        try:
            # Get local IP address
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)

            # Build service properties
            properties = self._build_service_properties()

            # Create service info
            self.service_info = ServiceInfo(
                type_=self.SERVICE_TYPE,
                name=f"{self.instance_name}.{self.SERVICE_TYPE}",
                addresses=[socket.inet_aton(local_ip)],
                port=self.port,
                properties=properties,
                server=f"{hostname}.local.",
            )

            # Register with Zeroconf
            self.zeroconf.register_service(self.service_info)

            self.file_logger.log_info(
                f"Registered service: {self.instance_name} at {local_ip}:{self.port}"
            )

        except Exception as e:
            self.file_logger.log_error(f"Failed to register service: {e}")
            raise

    def _build_service_properties(self) -> Dict[bytes, bytes]:
        """Build service advertisement properties."""
        model_config = self.config.get("model", {})
        network_config = self.config.get("network", {})
        resource_sharing = network_config.get("resource_sharing", {})

        # Convert properties to bytes (required by Zeroconf)
        properties = {
            b"instance_id": self.instance_id.encode("utf-8"),
            b"version": b"5.0.0",
            b"protocol": b"1.0",
            # Capabilities
            b"share_llm": str(resource_sharing.get("share_llm", True)).encode("utf-8"),
            b"share_embeddings": str(
                resource_sharing.get("share_embeddings", True)
            ).encode("utf-8"),
            b"share_memory": str(resource_sharing.get("share_memory", False)).encode(
                "utf-8"
            ),
            # Hardware info
            b"gpu_layers": str(model_config.get("gpu_layers", 0)).encode("utf-8"),
            b"context_size": str(model_config.get("context_size", 4096)).encode(
                "utf-8"
            ),
            # Status
            b"timestamp": str(int(time.time())).encode("utf-8"),
        }

        return properties

    def _on_service_added(self, info: ServiceInfo):
        """Handle discovery of a new peer."""
        try:
            # Parse service info
            peer_info = self._parse_service_info(info)

            # Don't add self
            if peer_info.instance_id == self.instance_id:
                return

            with self.peers_lock:
                self.peers[peer_info.instance_id] = peer_info

            self.file_logger.log_info(f"Discovered new peer: {peer_info}")

            if self.ui_logger:
                self.ui_logger.info(f"Peer discovered: {peer_info.instance_name}")

            # Trigger callbacks
            for callback in self.on_peer_added_callbacks:
                try:
                    callback(peer_info)
                except Exception as e:
                    self.file_logger.log_error(f"Error in peer_added callback: {e}")

        except Exception as e:
            self.file_logger.log_error(f"Error adding service: {e}")

    def _on_service_removed(self, name: str):
        """Handle removal of a peer."""
        try:
            # Find peer by service name
            peer_to_remove = None
            with self.peers_lock:
                for peer_id, peer in list(self.peers.items()):
                    if name.startswith(peer.instance_name):
                        peer_to_remove = peer
                        del self.peers[peer_id]
                        break

            if peer_to_remove:
                self.file_logger.log_info(f"Peer removed: {peer_to_remove}")

                if self.ui_logger:
                    self.ui_logger.warning(
                        f"Peer disconnected: {peer_to_remove.instance_name}"
                    )

                # Trigger callbacks
                for callback in self.on_peer_removed_callbacks:
                    try:
                        callback(peer_to_remove)
                    except Exception as e:
                        self.file_logger.log_error(
                            f"Error in peer_removed callback: {e}"
                        )

        except Exception as e:
            self.file_logger.log_error(f"Error removing service: {e}")

    def _on_service_updated(self, info: ServiceInfo):
        """Handle update of a peer."""
        try:
            peer_info = self._parse_service_info(info)

            # Don't update self
            if peer_info.instance_id == self.instance_id:
                return

            with self.peers_lock:
                if peer_info.instance_id in self.peers:
                    self.peers[peer_info.instance_id] = peer_info

            # Trigger callbacks
            for callback in self.on_peer_updated_callbacks:
                try:
                    callback(peer_info)
                except Exception as e:
                    self.file_logger.log_error(f"Error in peer_updated callback: {e}")

        except Exception as e:
            self.file_logger.log_error(f"Error updating service: {e}")

    def _parse_service_info(self, info: ServiceInfo) -> PeerInfo:
        """Parse Zeroconf ServiceInfo into PeerInfo."""
        # Extract address
        if info.addresses:
            address = socket.inet_ntoa(info.addresses[0])
        else:
            address = "unknown"

        # Decode properties
        properties = {}
        if info.properties:
            for key, value in info.properties.items():
                try:
                    properties[key.decode("utf-8")] = value.decode("utf-8")
                except Exception as e:
                    # Value is not UTF-8 decodable, convert to string
                    properties[key.decode("utf-8")] = str(value)

        # Extract instance ID and name
        instance_id = properties.get("instance_id", str(uuid.uuid4()))
        instance_name = info.name.split(".")[0]  # Extract name from full service name

        return PeerInfo(
            instance_id=instance_id,
            instance_name=instance_name,
            address=address,
            port=info.port,
            properties=properties,
            last_seen=time.time(),
        )

    def _health_monitor_loop(self):
        """Background thread to monitor peer health."""
        while self.running:
            try:
                current_time = time.time()

                with self.peers_lock:
                    # Remove stale peers (not seen in 3x TTL)
                    stale_threshold = current_time - (self.ttl * 3)
                    stale_peers = [
                        peer_id
                        for peer_id, peer in self.peers.items()
                        if peer.last_seen < stale_threshold
                    ]

                    for peer_id in stale_peers:
                        peer = self.peers[peer_id]
                        del self.peers[peer_id]
                        self.file_logger.log_warning(
                            f"Removed stale peer: {peer} (not seen for {self.ttl * 3}s)"
                        )

                # Sleep until next check
                time.sleep(self.health_check_interval)

            except Exception as e:
                self.file_logger.log_error(f"Error in health monitor: {e}")
                time.sleep(5)  # Brief pause before retry

    def get_peers(self) -> List[PeerInfo]:
        """Get list of currently discovered peers."""
        with self.peers_lock:
            return list(self.peers.values())

    def get_peer_count(self) -> int:
        """Get count of discovered peers."""
        with self.peers_lock:
            return len(self.peers)

    def get_peer_by_id(self, peer_id: str) -> Optional[PeerInfo]:
        """Get specific peer by ID."""
        with self.peers_lock:
            return self.peers.get(peer_id)

    def register_peer_added_callback(self, callback: Callable):
        """Register callback for when a peer is added."""
        self.on_peer_added_callbacks.append(callback)

    def register_peer_removed_callback(self, callback: Callable):
        """Register callback for when a peer is removed."""
        self.on_peer_removed_callbacks.append(callback)

    def register_peer_updated_callback(self, callback: Callable):
        """Register callback for when a peer is updated."""
        self.on_peer_updated_callbacks.append(callback)

    def get_status(self) -> Dict:
        """Get discovery service status."""
        with self.peers_lock:
            return {
                "instance_id": self.instance_id,
                "instance_name": self.instance_name,
                "port": self.port,
                "peer_count": len(self.peers),
                "peers": [peer.to_dict() for peer in self.peers.values()],
                "running": self.running,
            }
