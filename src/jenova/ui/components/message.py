##Script function and purpose: Styled message widgets for JENOVA TUI chat display
"""
Message components for JENOVA TUI.

Provides styled widgets for displaying user, AI, and system messages
with consistent visual hierarchy and color coding.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from textual.widgets import Static

if TYPE_CHECKING:
    pass


##Class purpose: Enum defining message sender types for styling
class MessageType(Enum):
    """Types of messages in the chat interface."""

    USER = "user"
    AI = "ai"
    SYSTEM = "system"
    ERROR = "error"
    INFO = "info"


##Class purpose: Styled chat message widget with sender-based coloring
class ChatMessage(Static):
    """
    A styled chat message widget.

    Displays messages with appropriate styling based on sender type.
    User messages are green, AI messages are cyan, system messages
    are yellow, and error messages are red.
    """

    ##Step purpose: Define component-specific CSS with entry animations
    DEFAULT_CSS = """
    ChatMessage {
        width: 100%;
        padding: 0 1;
        margin: 0 0 1 0;
        /* Animation purpose: Smooth fade-in for new messages */
        opacity: 1;
        transition: opacity 150ms ease-in;
    }

    ChatMessage.-new {
        /* Animation purpose: Start state for new message animation */
        opacity: 0;
    }

    ChatMessage.user {
        color: $success;
    }

    ChatMessage.user .message-prefix {
        color: $success-darken-1;
    }

    ChatMessage.ai {
        color: $primary-lighten-1;
    }

    ChatMessage.ai .message-prefix {
        color: $primary;
    }

    ChatMessage.system {
        color: $warning;
    }

    ChatMessage.system .message-prefix {
        color: $warning-darken-1;
    }

    ChatMessage.error {
        color: $error;
    }

    ChatMessage.error .message-prefix {
        color: $error-darken-1;
    }

    ChatMessage.info {
        color: $text-muted;
    }

    ChatMessage.info .message-prefix {
        color: $text-muted;
    }
    """

    ##Method purpose: Initialize message with content and type
    def __init__(
        self,
        content: str,
        message_type: MessageType = MessageType.USER,
        sender_name: str | None = None,
        **kwargs: object,
    ) -> None:
        """
        Initialize a chat message.

        Args:
            content: The message text content.
            message_type: Type of message for styling.
            sender_name: Optional custom sender name.
            **kwargs: Additional arguments passed to Static.
        """
        self._content = content
        self._message_type = message_type
        self._sender_name = sender_name

        ##Step purpose: Build formatted message with prefix
        formatted = self._format_message()

        super().__init__(formatted, **kwargs)

        ##Action purpose: Add CSS class based on message type
        self.add_class(message_type.value)

        ##Animation purpose: Start with hidden state for fade-in effect
        self.add_class("-new")

    ##Method purpose: Trigger fade-in animation after mount
    def on_mount(self) -> None:
        """Remove -new class after mount to trigger fade-in."""
        ##Animation purpose: Remove hidden class to trigger CSS transition
        self.remove_class("-new")

    ##Method purpose: Format message with appropriate prefix based on type
    def _format_message(self) -> str:
        """Format the message with sender prefix."""
        ##Step purpose: Determine prefix based on message type
        prefixes = {
            MessageType.USER: self._sender_name or "You",
            MessageType.AI: "JENOVA",
            MessageType.SYSTEM: ">>",
            MessageType.ERROR: "ERROR",
            MessageType.INFO: ">>",
        }

        prefix = prefixes.get(self._message_type, "")

        ##Condition purpose: Format differently for system/info messages
        if self._message_type in (MessageType.SYSTEM, MessageType.INFO):
            return f"{prefix} {self._content}"

        ##Step purpose: Format with bold prefix for chat messages
        return f"[bold]{prefix}:[/bold] {self._content}"

    ##Method purpose: Get the message type
    @property
    def message_type(self) -> MessageType:
        """Get the message type."""
        return self._message_type

    ##Method purpose: Get the raw content
    @property
    def content(self) -> str:
        """Get the raw message content."""
        return self._content


##Class purpose: Container for displaying a conversation thread
class MessageThread(Static):
    """
    A container for a series of messages.

    Provides consistent spacing and layout for message display.
    """

    ##Step purpose: Define thread container CSS
    DEFAULT_CSS = """
    MessageThread {
        width: 100%;
        height: auto;
        padding: 1;
    }
    """

    ##Method purpose: Initialize empty message thread
    def __init__(self, **kwargs: object) -> None:
        """Initialize an empty message thread."""
        super().__init__(**kwargs)
        self._messages: list[ChatMessage] = []

    ##Method purpose: Add a message to the thread
    def add_message(
        self,
        content: str,
        message_type: MessageType = MessageType.USER,
        sender_name: str | None = None,
    ) -> ChatMessage:
        """
        Add a new message to the thread.

        Args:
            content: The message text.
            message_type: Type of message for styling.
            sender_name: Optional custom sender name.

        Returns:
            The created ChatMessage widget.
        """
        ##Step purpose: Create and mount new message
        message = ChatMessage(
            content=content,
            message_type=message_type,
            sender_name=sender_name,
        )
        self._messages.append(message)
        self.mount(message)
        return message

    ##Method purpose: Clear all messages from thread
    def clear_messages(self) -> None:
        """Remove all messages from the thread."""
        ##Loop purpose: Remove each message widget
        for message in self._messages:
            message.remove()
        self._messages.clear()

    ##Method purpose: Get count of messages
    @property
    def message_count(self) -> int:
        """Get the number of messages in the thread."""
        return len(self._messages)
