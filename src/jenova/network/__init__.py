# The JENOVA Cognitive Architecture - Network Layer
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Network Layer for Distributed JENOVA

This package enables LAN-based discovery and distributed computing across
multiple JENOVA instances, allowing hardware resource pooling and federated
cognitive operations.

Core Components:
- Discovery: mDNS/Zeroconf-based peer discovery
- Peer Management: Connection lifecycle and capability tracking
- RPC Service: gRPC-based remote procedure calls
- Security: Certificate-based authentication and encryption
"""

from jenova.network.discovery import JenovaDiscoveryService
from jenova.network.peer_manager import PeerManager
from jenova.network.rpc_client import JenovaRPCClient
from jenova.network.security import SecurityManager

__all__ = [
    "JenovaDiscoveryService",
    "PeerManager",
    "JenovaRPCClient",
    "SecurityManager",
]

__version__ = "5.0.0"
__phase__ = "Phase 8: Distributed Computing & LAN Networking"
