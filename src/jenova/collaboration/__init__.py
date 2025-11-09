# The JENOVA Cognitive Architecture - Collaboration Module
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 30: Multi-User Collaboration - LAN-based collaborative sessions.

Implements multi-user collaboration with user session management,
shared conversation state, synchronization protocol, and access control.
100% offline, works over LAN, no external APIs.

Example:
    >>> from jenova.collaboration import CollaborationManager, UserRole
    >>> manager = CollaborationManager()
    >>> session = manager.create_session("alice", "Team Discussion")
    >>> manager.add_participant(session.session_id, "bob_session", UserRole.CONTRIBUTOR)
"""

# User session management
from jenova.collaboration.user_session import (
    UserRole,
    SessionStatus,
    UserProfile,
    UserSession,
    UserSessionManager,
)

# Collaboration management
from jenova.collaboration.collaboration_manager import (
    CollaborationMode,
    ConversationState,
    CollaborativeSession,
    CollaborationManager,
)

# Synchronization protocol
from jenova.collaboration.sync_protocol import (
    MessageType,
    SyncMessage,
    SyncProtocol,
    SyncClient,
)

# Access control
from jenova.collaboration.access_control import (
    Permission,
    AccessPolicy,
    AccessController,
    ROLE_PERMISSIONS,
)

__all__ = [
    # User session
    "UserRole",
    "SessionStatus",
    "UserProfile",
    "UserSession",
    "UserSessionManager",
    # Collaboration
    "CollaborationMode",
    "ConversationState",
    "CollaborativeSession",
    "CollaborationManager",
    # Synchronization
    "MessageType",
    "SyncMessage",
    "SyncProtocol",
    "SyncClient",
    # Access control
    "Permission",
    "AccessPolicy",
    "AccessController",
    "ROLE_PERMISSIONS",
]
