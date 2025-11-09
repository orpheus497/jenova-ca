# The JENOVA Cognitive Architecture - Synchronization Protocol
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 30: Synchronization Protocol - LAN-based state synchronization.

Implements message-based protocol for synchronizing collaborative
sessions across LAN. 100% offline, no external APIs.
"""

import time
import json
import socket
import threading
import queue
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class MessageType(str, Enum):
    """Synchronization message types."""
    # Connection
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    HEARTBEAT = "heartbeat"

    # Session management
    SESSION_CREATE = "session_create"
    SESSION_JOIN = "session_join"
    SESSION_LEAVE = "session_leave"
    SESSION_UPDATE = "session_update"

    # Conversation
    TURN_ADD = "turn_add"
    TURN_UPDATE = "turn_update"
    TURN_DELETE = "turn_delete"

    # Collaboration
    REQUEST_TURN = "request_turn"
    RELEASE_TURN = "release_turn"
    GRANT_TURN = "grant_turn"

    # Moderation
    CONTRIBUTION_SUBMIT = "contribution_submit"
    CONTRIBUTION_APPROVE = "contribution_approve"
    CONTRIBUTION_REJECT = "contribution_reject"

    # State sync
    STATE_REQUEST = "state_request"
    STATE_RESPONSE = "state_response"

    # Error
    ERROR = "error"


@dataclass
class SyncMessage:
    """
    Synchronization message.

    Attributes:
        message_id: Unique message identifier
        message_type: Message type
        sender_id: Sender's session ID
        timestamp: Message timestamp
        session_id: Target collaborative session ID
        payload: Message data
    """
    message_id: str
    message_type: MessageType
    sender_id: str
    timestamp: float
    session_id: str
    payload: Dict[str, Any]


class SyncProtocol:
    """
    LAN-based synchronization protocol.

    Implements UDP-based message broadcasting for local network
    collaboration. All communication stays on LAN, 100% offline.

    Example:
        >>> protocol = SyncProtocol(port=5000)
        >>> protocol.start()
        >>> protocol.send_message(message)
        >>> protocol.stop()
    """

    def __init__(
        self,
        port: int = 5000,
        broadcast_address: str = "255.255.255.255"
    ):
        """
        Initialize sync protocol.

        Args:
            port: UDP port for communication
            broadcast_address: Broadcast address for LAN
        """
        self.port = port
        self.broadcast_address = broadcast_address

        # Socket for sending/receiving
        self.socket: Optional[socket.socket] = None

        # Message handlers (message_type -> callback)
        self.handlers: Dict[MessageType, List[Callable]] = {}

        # Outgoing message queue
        self.outgoing_queue: queue.Queue = queue.Queue()

        # Running flag
        self.running = False

        # Threads
        self.send_thread: Optional[threading.Thread] = None
        self.receive_thread: Optional[threading.Thread] = None

        # Thread lock
        self._lock = threading.RLock()

    def start(self) -> bool:
        """
        Start synchronization protocol.

        Returns:
            True if started successfully

        Example:
            >>> protocol.start()
        """
        with self._lock:
            if self.running:
                return False

            try:
                # Create UDP socket
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.socket.bind(("", self.port))
                self.socket.settimeout(1.0)  # 1 second timeout

                self.running = True

                # Start threads
                self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
                self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)

                self.send_thread.start()
                self.receive_thread.start()

                return True

            except Exception:
                self.running = False
                if self.socket:
                    self.socket.close()
                return False

    def stop(self) -> None:
        """
        Stop synchronization protocol.

        Example:
            >>> protocol.stop()
        """
        with self._lock:
            if not self.running:
                return

            self.running = False

            # Wait for threads to finish
            if self.send_thread and self.send_thread.is_alive():
                self.send_thread.join(timeout=2.0)
            if self.receive_thread and self.receive_thread.is_alive():
                self.receive_thread.join(timeout=2.0)

            # Close socket
            if self.socket:
                self.socket.close()
                self.socket = None

    def send_message(self, message: SyncMessage) -> bool:
        """
        Send synchronization message.

        Args:
            message: SyncMessage to send

        Returns:
            True if queued for sending

        Example:
            >>> protocol.send_message(sync_message)
        """
        if not self.running:
            return False

        self.outgoing_queue.put(message)
        return True

    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[SyncMessage], None]
    ) -> None:
        """
        Register message handler.

        Args:
            message_type: Message type to handle
            handler: Callback function

        Example:
            >>> protocol.register_handler(MessageType.TURN_ADD, handle_turn)
        """
        with self._lock:
            if message_type not in self.handlers:
                self.handlers[message_type] = []
            self.handlers[message_type].append(handler)

    def unregister_handler(
        self,
        message_type: MessageType,
        handler: Callable[[SyncMessage], None]
    ) -> None:
        """
        Unregister message handler.

        Args:
            message_type: Message type
            handler: Callback function to remove

        Example:
            >>> protocol.unregister_handler(MessageType.TURN_ADD, handle_turn)
        """
        with self._lock:
            if message_type in self.handlers:
                self.handlers[message_type] = [
                    h for h in self.handlers[message_type]
                    if h != handler
                ]

    def _send_loop(self) -> None:
        """Send loop thread function."""
        while self.running:
            try:
                # Get message from queue (with timeout)
                try:
                    message = self.outgoing_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Serialize message
                data = json.dumps({
                    "message_id": message.message_id,
                    "message_type": message.message_type.value,
                    "sender_id": message.sender_id,
                    "timestamp": message.timestamp,
                    "session_id": message.session_id,
                    "payload": message.payload,
                }).encode("utf-8")

                # Send via UDP broadcast
                if self.socket:
                    self.socket.sendto(
                        data,
                        (self.broadcast_address, self.port)
                    )

            except Exception:
                # Continue on error
                pass

    def _receive_loop(self) -> None:
        """Receive loop thread function."""
        while self.running:
            try:
                if not self.socket:
                    break

                # Receive data
                try:
                    data, addr = self.socket.recvfrom(65536)
                except socket.timeout:
                    continue

                # Deserialize message
                message_data = json.loads(data.decode("utf-8"))

                message = SyncMessage(
                    message_id=message_data["message_id"],
                    message_type=MessageType(message_data["message_type"]),
                    sender_id=message_data["sender_id"],
                    timestamp=message_data["timestamp"],
                    session_id=message_data["session_id"],
                    payload=message_data["payload"],
                )

                # Dispatch to handlers
                self._dispatch_message(message)

            except Exception:
                # Continue on error
                pass

    def _dispatch_message(self, message: SyncMessage) -> None:
        """
        Dispatch message to registered handlers.

        Args:
            message: Received message
        """
        with self._lock:
            handlers = self.handlers.get(message.message_type, [])
            for handler in handlers:
                try:
                    handler(message)
                except Exception:
                    # Continue on handler error
                    pass


class SyncClient:
    """
    Synchronization client for managing protocol interactions.

    Provides higher-level API for sending specific message types
    and handling responses.

    Example:
        >>> client = SyncClient(session_id="alice_session")
        >>> client.start()
        >>> client.broadcast_turn_add(collab_session_id, turn_data)
    """

    def __init__(
        self,
        session_id: str,
        port: int = 5000,
        broadcast_address: str = "255.255.255.255"
    ):
        """
        Initialize sync client.

        Args:
            session_id: This user's session ID
            port: UDP port
            broadcast_address: Broadcast address
        """
        self.session_id = session_id
        self.protocol = SyncProtocol(port, broadcast_address)

        # Message counter for unique IDs
        self._message_counter = 0
        self._lock = threading.Lock()

    def start(self) -> bool:
        """
        Start sync client.

        Returns:
            True if started

        Example:
            >>> client.start()
        """
        return self.protocol.start()

    def stop(self) -> None:
        """
        Stop sync client.

        Example:
            >>> client.stop()
        """
        self.protocol.stop()

    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[SyncMessage], None]
    ) -> None:
        """
        Register message handler.

        Args:
            message_type: Message type
            handler: Callback function

        Example:
            >>> client.register_handler(MessageType.TURN_ADD, handle_turn)
        """
        self.protocol.register_handler(message_type, handler)

    def broadcast_turn_add(
        self,
        collab_session_id: str,
        turn_data: Dict[str, Any]
    ) -> bool:
        """
        Broadcast turn addition.

        Args:
            collab_session_id: Collaborative session ID
            turn_data: Turn data

        Returns:
            True if broadcast

        Example:
            >>> client.broadcast_turn_add(session_id, turn_data)
        """
        message = self._create_message(
            MessageType.TURN_ADD,
            collab_session_id,
            turn_data
        )
        return self.protocol.send_message(message)

    def broadcast_session_update(
        self,
        collab_session_id: str,
        update_data: Dict[str, Any]
    ) -> bool:
        """
        Broadcast session update.

        Args:
            collab_session_id: Collaborative session ID
            update_data: Update data

        Returns:
            True if broadcast

        Example:
            >>> client.broadcast_session_update(session_id, {"state": "locked"})
        """
        message = self._create_message(
            MessageType.SESSION_UPDATE,
            collab_session_id,
            update_data
        )
        return self.protocol.send_message(message)

    def request_turn(
        self,
        collab_session_id: str
    ) -> bool:
        """
        Request turn to contribute.

        Args:
            collab_session_id: Collaborative session ID

        Returns:
            True if request sent

        Example:
            >>> client.request_turn(session_id)
        """
        message = self._create_message(
            MessageType.REQUEST_TURN,
            collab_session_id,
            {}
        )
        return self.protocol.send_message(message)

    def send_heartbeat(
        self,
        collab_session_id: str
    ) -> bool:
        """
        Send heartbeat message.

        Args:
            collab_session_id: Collaborative session ID

        Returns:
            True if sent

        Example:
            >>> client.send_heartbeat(session_id)
        """
        message = self._create_message(
            MessageType.HEARTBEAT,
            collab_session_id,
            {"timestamp": time.time()}
        )
        return self.protocol.send_message(message)

    def _create_message(
        self,
        message_type: MessageType,
        collab_session_id: str,
        payload: Dict[str, Any]
    ) -> SyncMessage:
        """
        Create sync message.

        Args:
            message_type: Message type
            collab_session_id: Session ID
            payload: Message payload

        Returns:
            SyncMessage
        """
        with self._lock:
            self._message_counter += 1
            message_id = f"{self.session_id}_{self._message_counter}"

        return SyncMessage(
            message_id=message_id,
            message_type=message_type,
            sender_id=self.session_id,
            timestamp=time.time(),
            session_id=collab_session_id,
            payload=payload,
        )
