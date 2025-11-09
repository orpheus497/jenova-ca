# The JENOVA Cognitive Architecture - User Session Management
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 30: User Session Management - Track and manage user sessions.

Handles user identification, session lifecycle, activity tracking,
and user metadata. 100% offline, works over LAN.
"""

import time
import uuid
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime


class UserRole(str, Enum):
    """User role in collaboration session."""
    OWNER = "owner"  # Session creator, full control
    ADMIN = "admin"  # Administrative privileges
    CONTRIBUTOR = "contributor"  # Can add content
    VIEWER = "viewer"  # Read-only access


class SessionStatus(str, Enum):
    """User session status."""
    ACTIVE = "active"
    IDLE = "idle"
    DISCONNECTED = "disconnected"
    KICKED = "kicked"


@dataclass
class UserProfile:
    """
    User profile information.

    Attributes:
        user_id: Unique user identifier
        username: Display name
        display_color: Color for UI (hex code)
        created_at: Account creation timestamp
        metadata: Additional user metadata
    """
    user_id: str
    username: str
    display_color: str = "#3498db"
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserSession:
    """
    Active user session in collaboration.

    Attributes:
        session_id: Unique session identifier
        user_profile: User profile information
        role: User's role in this session
        status: Current session status
        joined_at: Session join timestamp
        last_activity: Last activity timestamp
        ip_address: User's IP address (LAN)
        turns_contributed: Number of conversation turns added
        messages_sent: Number of messages sent
        metadata: Additional session metadata
    """
    session_id: str
    user_profile: UserProfile
    role: UserRole
    status: SessionStatus = SessionStatus.ACTIVE
    joined_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    ip_address: str = "127.0.0.1"
    turns_contributed: int = 0
    messages_sent: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class UserSessionManager:
    """
    Manage user sessions in collaboration.

    Tracks active users, manages session lifecycle, and maintains
    user profiles and activity metrics.

    Example:
        >>> manager = UserSessionManager()
        >>> profile = manager.create_user("alice", "Alice")
        >>> session = manager.create_session(profile, UserRole.OWNER)
        >>> manager.update_activity(session.session_id)
    """

    def __init__(self):
        """Initialize user session manager."""
        # User profiles (user_id -> UserProfile)
        self.profiles: Dict[str, UserProfile] = {}

        # Active sessions (session_id -> UserSession)
        self.sessions: Dict[str, UserSession] = {}

        # User to session mapping (user_id -> set of session_ids)
        self.user_sessions: Dict[str, Set[str]] = {}

    def create_user(
        self,
        user_id: str,
        username: str,
        display_color: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UserProfile:
        """
        Create new user profile.

        Args:
            user_id: Unique user identifier
            username: Display name
            display_color: UI color (hex code)
            metadata: Additional user metadata

        Returns:
            Created UserProfile

        Example:
            >>> profile = manager.create_user("alice", "Alice", "#ff5733")
        """
        if user_id in self.profiles:
            return self.profiles[user_id]

        profile = UserProfile(
            user_id=user_id,
            username=username,
            display_color=display_color or self._generate_color(),
            metadata=metadata or {}
        )

        self.profiles[user_id] = profile
        self.user_sessions[user_id] = set()

        return profile

    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """
        Get user profile by ID.

        Args:
            user_id: User identifier

        Returns:
            UserProfile or None

        Example:
            >>> profile = manager.get_user("alice")
        """
        return self.profiles.get(user_id)

    def update_user(
        self,
        user_id: str,
        username: Optional[str] = None,
        display_color: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update user profile.

        Args:
            user_id: User identifier
            username: New username
            display_color: New color
            metadata: Metadata updates

        Returns:
            True if updated

        Example:
            >>> manager.update_user("alice", username="Alice Smith")
        """
        if user_id not in self.profiles:
            return False

        profile = self.profiles[user_id]

        if username is not None:
            profile.username = username
        if display_color is not None:
            profile.display_color = display_color
        if metadata is not None:
            profile.metadata.update(metadata)

        return True

    def create_session(
        self,
        user_profile: UserProfile,
        role: UserRole,
        ip_address: str = "127.0.0.1",
        metadata: Optional[Dict[str, Any]] = None
    ) -> UserSession:
        """
        Create new user session.

        Args:
            user_profile: User profile
            role: User's role
            ip_address: User's IP address
            metadata: Additional session metadata

        Returns:
            Created UserSession

        Example:
            >>> session = manager.create_session(profile, UserRole.CONTRIBUTOR)
        """
        session_id = str(uuid.uuid4())

        session = UserSession(
            session_id=session_id,
            user_profile=user_profile,
            role=role,
            ip_address=ip_address,
            metadata=metadata or {}
        )

        self.sessions[session_id] = session
        self.user_sessions[user_profile.user_id].add(session_id)

        return session

    def get_session(self, session_id: str) -> Optional[UserSession]:
        """
        Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            UserSession or None

        Example:
            >>> session = manager.get_session(session_id)
        """
        return self.sessions.get(session_id)

    def end_session(self, session_id: str) -> bool:
        """
        End user session.

        Args:
            session_id: Session identifier

        Returns:
            True if ended

        Example:
            >>> manager.end_session(session_id)
        """
        session = self.sessions.get(session_id)
        if not session:
            return False

        session.status = SessionStatus.DISCONNECTED

        # Remove from user's session set
        self.user_sessions[session.user_profile.user_id].discard(session_id)

        # Remove from active sessions
        del self.sessions[session_id]

        return True

    def update_activity(
        self,
        session_id: str,
        turn_contributed: bool = False,
        message_sent: bool = False
    ) -> bool:
        """
        Update session activity timestamp.

        Args:
            session_id: Session identifier
            turn_contributed: User added conversation turn
            message_sent: User sent message

        Returns:
            True if updated

        Example:
            >>> manager.update_activity(session_id, turn_contributed=True)
        """
        session = self.sessions.get(session_id)
        if not session:
            return False

        session.last_activity = time.time()

        if turn_contributed:
            session.turns_contributed += 1
        if message_sent:
            session.messages_sent += 1

        # Update status to active if idle
        if session.status == SessionStatus.IDLE:
            session.status = SessionStatus.ACTIVE

        return True

    def set_session_status(
        self,
        session_id: str,
        status: SessionStatus
    ) -> bool:
        """
        Set session status.

        Args:
            session_id: Session identifier
            status: New status

        Returns:
            True if updated

        Example:
            >>> manager.set_session_status(session_id, SessionStatus.IDLE)
        """
        session = self.sessions.get(session_id)
        if not session:
            return False

        session.status = status
        return True

    def get_active_sessions(self) -> List[UserSession]:
        """
        Get all active sessions.

        Returns:
            List of active UserSession objects

        Example:
            >>> active = manager.get_active_sessions()
        """
        return [
            session for session in self.sessions.values()
            if session.status == SessionStatus.ACTIVE
        ]

    def get_user_sessions(self, user_id: str) -> List[UserSession]:
        """
        Get all sessions for user.

        Args:
            user_id: User identifier

        Returns:
            List of UserSession objects

        Example:
            >>> sessions = manager.get_user_sessions("alice")
        """
        session_ids = self.user_sessions.get(user_id, set())
        return [
            self.sessions[sid]
            for sid in session_ids
            if sid in self.sessions
        ]

    def check_idle_sessions(self, idle_timeout: int = 300) -> List[str]:
        """
        Check for idle sessions.

        Args:
            idle_timeout: Idle timeout in seconds (default: 5 minutes)

        Returns:
            List of idle session IDs

        Example:
            >>> idle_sessions = manager.check_idle_sessions(idle_timeout=600)
        """
        now = time.time()
        idle_sessions = []

        for session_id, session in self.sessions.items():
            if session.status != SessionStatus.ACTIVE:
                continue

            if now - session.last_activity > idle_timeout:
                session.status = SessionStatus.IDLE
                idle_sessions.append(session_id)

        return idle_sessions

    def kick_session(self, session_id: str, reason: str = "") -> bool:
        """
        Kick user from session.

        Args:
            session_id: Session identifier
            reason: Kick reason

        Returns:
            True if kicked

        Example:
            >>> manager.kick_session(session_id, "Violation of terms")
        """
        session = self.sessions.get(session_id)
        if not session:
            return False

        session.status = SessionStatus.KICKED
        session.metadata["kick_reason"] = reason
        session.metadata["kicked_at"] = time.time()

        return True

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get session statistics.

        Args:
            session_id: Session identifier

        Returns:
            Statistics dictionary

        Example:
            >>> stats = manager.get_session_stats(session_id)
        """
        session = self.sessions.get(session_id)
        if not session:
            return {}

        now = time.time()
        duration = now - session.joined_at
        idle_time = now - session.last_activity

        return {
            "session_id": session_id,
            "user_id": session.user_profile.user_id,
            "username": session.user_profile.username,
            "role": session.role.value,
            "status": session.status.value,
            "duration_seconds": duration,
            "idle_seconds": idle_time,
            "turns_contributed": session.turns_contributed,
            "messages_sent": session.messages_sent,
            "joined_at": session.joined_at,
            "last_activity": session.last_activity,
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get all session statistics.

        Returns:
            Comprehensive statistics

        Example:
            >>> stats = manager.get_all_stats()
        """
        active = [s for s in self.sessions.values() if s.status == SessionStatus.ACTIVE]
        idle = [s for s in self.sessions.values() if s.status == SessionStatus.IDLE]

        return {
            "total_users": len(self.profiles),
            "total_sessions": len(self.sessions),
            "active_sessions": len(active),
            "idle_sessions": len(idle),
            "total_turns_contributed": sum(s.turns_contributed for s in self.sessions.values()),
            "total_messages_sent": sum(s.messages_sent for s in self.sessions.values()),
            "sessions_by_role": {
                role.value: len([s for s in self.sessions.values() if s.role == role])
                for role in UserRole
            },
        }

    def _generate_color(self) -> str:
        """
        Generate random display color.

        Returns:
            Hex color code
        """
        import random
        colors = [
            "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
            "#1abc9c", "#34495e", "#e67e22", "#95a5a6", "#d35400"
        ]
        return random.choice(colors)
