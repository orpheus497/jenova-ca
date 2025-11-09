# The JENOVA Cognitive Architecture - Collaboration Manager
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 30: Collaboration Manager - Multi-user collaboration orchestration.

Manages shared conversation sessions, user interactions, turn-taking,
and synchronization. 100% offline, works over LAN.
"""

import time
import uuid
import threading
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import json

from jenova.collaboration.user_session import (
    UserSessionManager,
    UserProfile,
    UserSession,
    UserRole,
    SessionStatus,
)


class CollaborationMode(str, Enum):
    """Collaboration mode."""
    SEQUENTIAL = "sequential"  # Turn-based, one at a time
    CONCURRENT = "concurrent"  # Multiple users simultaneously
    MODERATED = "moderated"  # Requires approval from moderator


class ConversationState(str, Enum):
    """Shared conversation state."""
    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    WAITING_APPROVAL = "waiting_approval"
    LOCKED = "locked"


@dataclass
class CollaborativeSession:
    """
    Shared collaboration session.

    Attributes:
        session_id: Unique session identifier
        name: Session name
        description: Session description
        owner_id: Session owner user ID
        mode: Collaboration mode
        state: Current conversation state
        created_at: Creation timestamp
        last_activity: Last activity timestamp
        participants: Set of active session IDs
        conversation_turns: Shared conversation history
        pending_contributions: Pending contributions awaiting approval
        metadata: Additional session metadata
    """
    session_id: str
    name: str
    description: str
    owner_id: str
    mode: CollaborationMode
    state: ConversationState = ConversationState.IDLE
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    participants: Set[str] = field(default_factory=set)
    conversation_turns: List[Dict[str, Any]] = field(default_factory=list)
    pending_contributions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CollaborationManager:
    """
    Manage multi-user collaboration sessions.

    Orchestrates shared conversations, manages turn-taking, handles
    user contributions, and maintains synchronization across participants.

    Example:
        >>> manager = CollaborationManager()
        >>> session = manager.create_session("alice", "Brainstorm", "Team discussion")
        >>> manager.add_participant(session.session_id, "bob_session", UserRole.CONTRIBUTOR)
        >>> manager.add_turn(session.session_id, "alice_session", "What should we build?")
    """

    def __init__(self):
        """Initialize collaboration manager."""
        # User session manager
        self.user_manager = UserSessionManager()

        # Collaborative sessions (session_id -> CollaborativeSession)
        self.sessions: Dict[str, CollaborativeSession] = {}

        # Active turn locks (session_id -> user_session_id)
        self.turn_locks: Dict[str, str] = {}

        # Thread safety
        self._lock = threading.RLock()

    def create_session(
        self,
        owner_user_id: str,
        name: str,
        description: str = "",
        mode: CollaborationMode = CollaborationMode.SEQUENTIAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CollaborativeSession:
        """
        Create new collaborative session.

        Args:
            owner_user_id: Owner's user ID
            name: Session name
            description: Session description
            mode: Collaboration mode
            metadata: Additional metadata

        Returns:
            Created CollaborativeSession

        Example:
            >>> session = manager.create_session("alice", "Project Planning")
        """
        with self._lock:
            session_id = str(uuid.uuid4())

            session = CollaborativeSession(
                session_id=session_id,
                name=name,
                description=description,
                owner_id=owner_user_id,
                mode=mode,
                metadata=metadata or {}
            )

            self.sessions[session_id] = session

            return session

    def add_participant(
        self,
        collab_session_id: str,
        user_session_id: str,
        role: UserRole
    ) -> bool:
        """
        Add participant to collaborative session.

        Args:
            collab_session_id: Collaborative session ID
            user_session_id: User's session ID
            role: User's role in session

        Returns:
            True if added

        Example:
            >>> manager.add_participant(session_id, user_session_id, UserRole.CONTRIBUTOR)
        """
        with self._lock:
            session = self.sessions.get(collab_session_id)
            user_session = self.user_manager.get_session(user_session_id)

            if not session or not user_session:
                return False

            # Update user session role
            user_session.role = role

            # Add to participants
            session.participants.add(user_session_id)
            session.last_activity = time.time()

            return True

    def remove_participant(
        self,
        collab_session_id: str,
        user_session_id: str
    ) -> bool:
        """
        Remove participant from session.

        Args:
            collab_session_id: Collaborative session ID
            user_session_id: User's session ID

        Returns:
            True if removed

        Example:
            >>> manager.remove_participant(session_id, user_session_id)
        """
        with self._lock:
            session = self.sessions.get(collab_session_id)
            if not session:
                return False

            session.participants.discard(user_session_id)
            session.last_activity = time.time()

            # Release turn lock if held by this user
            if self.turn_locks.get(collab_session_id) == user_session_id:
                del self.turn_locks[collab_session_id]

            return True

    def request_turn(
        self,
        collab_session_id: str,
        user_session_id: str
    ) -> bool:
        """
        Request turn to contribute (sequential mode).

        Args:
            collab_session_id: Collaborative session ID
            user_session_id: User's session ID

        Returns:
            True if turn granted

        Example:
            >>> can_contribute = manager.request_turn(session_id, user_session_id)
        """
        with self._lock:
            session = self.sessions.get(collab_session_id)
            user_session = self.user_manager.get_session(user_session_id)

            if not session or not user_session:
                return False

            # Check mode
            if session.mode == CollaborationMode.CONCURRENT:
                # Always granted in concurrent mode
                return True

            # Check if someone else has turn
            current_holder = self.turn_locks.get(collab_session_id)
            if current_holder and current_holder != user_session_id:
                return False

            # Grant turn
            self.turn_locks[collab_session_id] = user_session_id
            return True

    def release_turn(
        self,
        collab_session_id: str,
        user_session_id: str
    ) -> bool:
        """
        Release turn (sequential mode).

        Args:
            collab_session_id: Collaborative session ID
            user_session_id: User's session ID

        Returns:
            True if released

        Example:
            >>> manager.release_turn(session_id, user_session_id)
        """
        with self._lock:
            session = self.sessions.get(collab_session_id)
            if not session:
                return False

            # Check if user holds turn
            if self.turn_locks.get(collab_session_id) != user_session_id:
                return False

            # Release turn
            del self.turn_locks[collab_session_id]
            return True

    def add_turn(
        self,
        collab_session_id: str,
        user_session_id: str,
        user_message: str,
        assistant_response: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Add conversation turn to session.

        Args:
            collab_session_id: Collaborative session ID
            user_session_id: User's session ID
            user_message: User's message
            assistant_response: Optional assistant response
            metadata: Additional metadata

        Returns:
            Turn ID if added, None otherwise

        Example:
            >>> turn_id = manager.add_turn(session_id, user_session_id, "Hello!")
        """
        with self._lock:
            session = self.sessions.get(collab_session_id)
            user_session = self.user_manager.get_session(user_session_id)

            if not session or not user_session:
                return None

            # Check permissions
            if user_session.role not in [UserRole.OWNER, UserRole.ADMIN, UserRole.CONTRIBUTOR]:
                return None

            # Check if moderated
            if session.mode == CollaborationMode.MODERATED:
                # Add to pending
                return self._add_pending_contribution(
                    session,
                    user_session_id,
                    user_message,
                    assistant_response,
                    metadata
                )

            # Check turn lock (sequential mode)
            if session.mode == CollaborationMode.SEQUENTIAL:
                if self.turn_locks.get(collab_session_id) != user_session_id:
                    return None

            # Add turn
            turn_id = str(uuid.uuid4())
            turn_data = {
                "turn_id": turn_id,
                "user_session_id": user_session_id,
                "user_id": user_session.user_profile.user_id,
                "username": user_session.user_profile.username,
                "timestamp": time.time(),
                "user_message": user_message,
                "assistant_response": assistant_response or "",
                "metadata": metadata or {}
            }

            session.conversation_turns.append(turn_data)
            session.last_activity = time.time()
            session.state = ConversationState.IN_PROGRESS

            # Update user activity
            self.user_manager.update_activity(user_session_id, turn_contributed=True)

            # Release turn in sequential mode
            if session.mode == CollaborationMode.SEQUENTIAL:
                self.release_turn(collab_session_id, user_session_id)

            return turn_id

    def _add_pending_contribution(
        self,
        session: CollaborativeSession,
        user_session_id: str,
        user_message: str,
        assistant_response: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Add contribution to pending queue."""
        contribution_id = str(uuid.uuid4())
        user_session = self.user_manager.get_session(user_session_id)

        contribution = {
            "contribution_id": contribution_id,
            "user_session_id": user_session_id,
            "user_id": user_session.user_profile.user_id if user_session else "unknown",
            "username": user_session.user_profile.username if user_session else "Unknown",
            "timestamp": time.time(),
            "user_message": user_message,
            "assistant_response": assistant_response or "",
            "metadata": metadata or {},
            "status": "pending"
        }

        session.pending_contributions.append(contribution)
        session.state = ConversationState.WAITING_APPROVAL

        return contribution_id

    def approve_contribution(
        self,
        collab_session_id: str,
        contribution_id: str,
        approver_session_id: str
    ) -> bool:
        """
        Approve pending contribution (moderated mode).

        Args:
            collab_session_id: Collaborative session ID
            contribution_id: Contribution ID to approve
            approver_session_id: Approver's session ID

        Returns:
            True if approved

        Example:
            >>> manager.approve_contribution(session_id, contrib_id, moderator_session_id)
        """
        with self._lock:
            session = self.sessions.get(collab_session_id)
            approver_session = self.user_manager.get_session(approver_session_id)

            if not session or not approver_session:
                return False

            # Check approver permissions
            if approver_session.role not in [UserRole.OWNER, UserRole.ADMIN]:
                return False

            # Find contribution
            contribution = None
            for c in session.pending_contributions:
                if c["contribution_id"] == contribution_id:
                    contribution = c
                    break

            if not contribution:
                return False

            # Convert to turn
            turn_id = str(uuid.uuid4())
            turn_data = {
                "turn_id": turn_id,
                "user_session_id": contribution["user_session_id"],
                "user_id": contribution["user_id"],
                "username": contribution["username"],
                "timestamp": time.time(),
                "user_message": contribution["user_message"],
                "assistant_response": contribution["assistant_response"],
                "metadata": contribution["metadata"],
                "approved_by": approver_session_id,
            }

            session.conversation_turns.append(turn_data)

            # Remove from pending
            session.pending_contributions.remove(contribution)

            # Update state
            if not session.pending_contributions:
                session.state = ConversationState.IN_PROGRESS

            return True

    def reject_contribution(
        self,
        collab_session_id: str,
        contribution_id: str,
        rejector_session_id: str,
        reason: str = ""
    ) -> bool:
        """
        Reject pending contribution.

        Args:
            collab_session_id: Collaborative session ID
            contribution_id: Contribution ID to reject
            rejector_session_id: Rejector's session ID
            reason: Rejection reason

        Returns:
            True if rejected

        Example:
            >>> manager.reject_contribution(session_id, contrib_id, moderator_session_id)
        """
        with self._lock:
            session = self.sessions.get(collab_session_id)
            rejector_session = self.user_manager.get_session(rejector_session_id)

            if not session or not rejector_session:
                return False

            # Check rejector permissions
            if rejector_session.role not in [UserRole.OWNER, UserRole.ADMIN]:
                return False

            # Find and remove contribution
            for c in session.pending_contributions:
                if c["contribution_id"] == contribution_id:
                    c["status"] = "rejected"
                    c["rejected_by"] = rejector_session_id
                    c["rejection_reason"] = reason
                    session.pending_contributions.remove(c)

                    # Update state
                    if not session.pending_contributions:
                        session.state = ConversationState.IN_PROGRESS

                    return True

            return False

    def get_conversation_history(
        self,
        collab_session_id: str,
        max_turns: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history.

        Args:
            collab_session_id: Collaborative session ID
            max_turns: Maximum turns to return

        Returns:
            List of conversation turns

        Example:
            >>> history = manager.get_conversation_history(session_id, max_turns=10)
        """
        with self._lock:
            session = self.sessions.get(collab_session_id)
            if not session:
                return []

            turns = session.conversation_turns
            if max_turns:
                turns = turns[-max_turns:]

            return turns

    def get_session_participants(
        self,
        collab_session_id: str
    ) -> List[UserSession]:
        """
        Get all session participants.

        Args:
            collab_session_id: Collaborative session ID

        Returns:
            List of UserSession objects

        Example:
            >>> participants = manager.get_session_participants(session_id)
        """
        with self._lock:
            session = self.sessions.get(collab_session_id)
            if not session:
                return []

            return [
                self.user_manager.get_session(sid)
                for sid in session.participants
                if self.user_manager.get_session(sid) is not None
            ]

    def save_session(
        self,
        collab_session_id: str,
        file_path: Path
    ) -> bool:
        """
        Save collaborative session to file.

        Args:
            collab_session_id: Collaborative session ID
            file_path: Output file path

        Returns:
            True if saved

        Example:
            >>> manager.save_session(session_id, Path("session.json"))
        """
        with self._lock:
            session = self.sessions.get(collab_session_id)
            if not session:
                return False

            data = {
                "session_id": session.session_id,
                "name": session.name,
                "description": session.description,
                "owner_id": session.owner_id,
                "mode": session.mode.value,
                "state": session.state.value,
                "created_at": session.created_at,
                "last_activity": session.last_activity,
                "participants": list(session.participants),
                "conversation_turns": session.conversation_turns,
                "pending_contributions": session.pending_contributions,
                "metadata": session.metadata,
            }

            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True

    def get_session_stats(
        self,
        collab_session_id: str
    ) -> Dict[str, Any]:
        """
        Get session statistics.

        Args:
            collab_session_id: Collaborative session ID

        Returns:
            Statistics dictionary

        Example:
            >>> stats = manager.get_session_stats(session_id)
        """
        with self._lock:
            session = self.sessions.get(collab_session_id)
            if not session:
                return {}

            participants = self.get_session_participants(collab_session_id)
            active_participants = [
                p for p in participants
                if p.status == SessionStatus.ACTIVE
            ]

            return {
                "session_id": collab_session_id,
                "name": session.name,
                "mode": session.mode.value,
                "state": session.state.value,
                "total_participants": len(session.participants),
                "active_participants": len(active_participants),
                "total_turns": len(session.conversation_turns),
                "pending_contributions": len(session.pending_contributions),
                "duration_seconds": time.time() - session.created_at,
                "created_at": session.created_at,
                "last_activity": session.last_activity,
            }
