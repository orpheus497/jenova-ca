##Script function and purpose: Insight Manager for The JENOVA Cognitive Architecture
"""
Insight Manager

Manages creation, storage, and retrieval of topical insights.
Insights are organized by concerns (topic categories) and linked
to the cognitive graph.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import structlog

from jenova.exceptions import InsightSaveError
from jenova.graph.types import Node
from jenova.insights.concerns import ConcernManager
from jenova.insights.types import CortexId, Insight, InsightSearchResult
from jenova.llm.types import GenerationParams
from jenova.utils.validation import (
    validate_username,
    validate_topic,
    validate_path_within_base,
)


##Step purpose: Get module logger
logger = structlog.get_logger(__name__)


##Class purpose: Protocol for cognitive graph operations
@runtime_checkable
class GraphProtocol(Protocol):
    """Protocol for cognitive graph dependency.
    
    Defines the minimal graph operations required by InsightManager.
    Implementations: CognitiveGraph (src/jenova/graph/graph.py)
    
    Contract:
        - add_node(node: Node) -> None: Must persist node to graph storage
        - has_node(node_id: str) -> bool: Must return True if node exists by ID
    """
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph.
        
        Args:
            node: Node object to add (must have valid id, label, content)
        
        Raises:
            GraphError: If node cannot be persisted
        """
        ...
    
    def has_node(self, node_id: str) -> bool:
        """Check if node exists in graph.
        
        Args:
            node_id: UUID string of the node
            
        Returns:
            True if node exists, False otherwise
        """
        ...


##Class purpose: Protocol for LLM operations
@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM dependency.
    
    Defines text generation interface for topic classification.
    Implementations: LLMInterface (src/jenova/llm/interface.py)
    
    Contract:
        - generate_text: Must return generated text as string
        - Must handle system_prompt for behavior control
        - Should respect params for generation settings
    """
    
    def generate_text(
        self,
        text: str,
        system_prompt: str = "You are a helpful assistant.",
        params: GenerationParams | None = None,
    ) -> str:
        """Generate text completion from prompt.
        
        Args:
            text: The input prompt text
            system_prompt: System message to control LLM behavior
            params: Optional generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated text response
            
        Raises:
            LLMError: If generation fails
        """
        ...


##Class purpose: Protocol for memory search operations
@runtime_checkable
class MemorySearchProtocol(Protocol):
    """Protocol for memory search dependency.
    
    Defines semantic search interface for insight retrieval.
    Implementations: Memory (src/jenova/memory/memory.py)
    
    Note: The actual Memory.search() returns list[MemoryResult], but this
    protocol expects list[tuple[str, float]] for backward compatibility.
    Callers should adapt the return type as needed.
    
    Contract:
        - search: Must return (content, distance) tuples sorted by relevance
        - Lower distance = more relevant
    """
    
    def search(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[tuple[str, float]]:
        """Search memory for semantically similar content.
        
        Args:
            query: Search query text
            n_results: Maximum number of results to return
            
        Returns:
            List of (content, distance) tuples, lower distance = more relevant
        """
        ...


##Class purpose: Manages lifecycle of insights including creation, storage, and retrieval
class InsightManager:
    """
    Manages the creation, storage, and retrieval of topical insights.
    
    Insights are organized hierarchically by user and topic/concern.
    Each insight is stored as a JSON file and linked to the cognitive graph.
    """
    
    ##Method purpose: Initialize insight manager with dependencies
    def __init__(
        self,
        insights_root: Path,
        graph: GraphProtocol,
        llm: LLMProtocol,
        memory_search: MemorySearchProtocol | None = None,
    ) -> None:
        """
        Initialize insight manager.
        
        Args:
            insights_root: Root directory for insight storage
            graph: Cognitive graph for node linking
            llm: LLM interface for topic classification
            memory_search: Optional memory search for semantic retrieval
        """
        ##Step purpose: Store dependencies
        self._insights_root = insights_root
        self._graph = graph
        self._llm = llm
        self._memory_search = memory_search
        
        ##Step purpose: Ensure root directory exists
        insights_root.mkdir(parents=True, exist_ok=True)
        
        ##Action purpose: Initialize concern manager
        self._concern_manager = ConcernManager(
            insights_root=insights_root,
            llm=llm,
        )
        
        logger.info(
            "insight_manager_initialized",
            storage_path=str(insights_root),
        )
    
    ##Method purpose: Save an insight, finding or creating a concern for it
    def save_insight(
        self,
        content: str,
        username: str,
        topic: str | None = None,
        linked_to: list[CortexId] | None = None,
        cortex_id: CortexId | None = None,
    ) -> Insight:
        """
        Save an insight.
        
        Finds or creates an appropriate concern/topic for the insight,
        creates a graph node, and persists to disk.
        
        Args:
            content: The insight content
            username: User this insight relates to
            topic: Optional topic (auto-classified if None)
            linked_to: Optional list of cortex IDs to link
            cortex_id: Optional existing cortex ID to update
            
        Returns:
            The saved Insight
            
        Raises:
            InsightSaveError: If save operation fails
        """
        logger.debug(
            "saving_insight",
            username=username,
            content_preview=content[:50],
        )
        
        ##Error purpose: Wrap any failures in InsightSaveError
        try:
            ##Step purpose: Validate username before use
            safe_username = validate_username(username)
            
            ##Step purpose: Determine topic using concern manager
            if not topic:
                existing_topics = self._concern_manager.get_all_concerns()
                topic = self._concern_manager.find_or_create_concern(
                    content, 
                    existing_topics,
                )
            
            ##Step purpose: Validate topic before use
            safe_topic = validate_topic(topic)
            
            ##Step purpose: Create or update graph node
            if cortex_id and self._graph.has_node(cortex_id):
                node_id = cortex_id
                logger.debug(
                    "updating_existing_node",
                    node_id=node_id,
                )
            else:
                node = Node.create(
                    label=content[:50],
                    content=content,
                    node_type="insight",
                    metadata={"username": safe_username, "topic": safe_topic},
                )
                self._graph.add_node(node)
                node_id = node.id
                logger.debug(
                    "created_insight_node",
                    node_id=node_id,
                )
            
            ##Step purpose: Create insight record
            insight = Insight(
                content=content,
                username=safe_username,
                topic=safe_topic,
                cortex_id=node_id,
                related_concerns=[],
            )
            
            ##Step purpose: Persist to file system
            self._save_insight_file(insight)
            
            logger.info(
                "insight_saved",
                username=username,
                topic=topic,
                cortex_id=node_id,
            )
            
            return insight
            
        except Exception as e:
            logger.error(
                "insight_save_failed",
                error=str(e),
                username=username,
            )
            raise InsightSaveError(content, str(e)) from e
    
    ##Method purpose: Save insight to filesystem as JSON
    def _save_insight_file(self, insight: Insight) -> Path:
        """
        Save insight to filesystem.
        
        Args:
            insight: Insight to save
            
        Returns:
            Path to saved file
            
        Raises:
            ValueError: If path traversal is detected
        """
        ##Step purpose: Validate username and topic (should already be validated, but double-check)
        safe_username = validate_username(insight.username)
        safe_topic = validate_topic(insight.topic)
        
        ##Step purpose: Build directory path with validated components
        user_dir = self._insights_root / safe_username
        topic_dir = user_dir / safe_topic
        
        ##Step purpose: Verify path is within base directory
        validate_path_within_base(topic_dir, self._insights_root)
        
        ##Action purpose: Create directory with secure permissions
        topic_dir.mkdir(parents=True, exist_ok=True)
        
        ##Step purpose: Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"insight_{timestamp}.json"
        filepath = topic_dir / filename
        
        ##Action purpose: Write insight JSON
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(insight.to_dict(), f, indent=4, ensure_ascii=False)
        
        ##Action purpose: Set secure file permissions (0o600 = owner read/write only)
        import os
        os.chmod(filepath, 0o600)
        
        return filepath
    
    ##Method purpose: Get relevant insights for a query using semantic search
    def get_relevant_insights(
        self,
        query: str,
        username: str,
        max_insights: int = 3,
    ) -> list[str]:
        """
        Get relevant insights for a query using semantic search.
        
        Args:
            query: Search query
            username: Username to filter insights for
            max_insights: Maximum insights to return
            
        Returns:
            List of insight content strings
        """
        ##Condition purpose: Return empty if no memory search
        if self._memory_search is None:
            logger.debug(
                "no_memory_search_configured",
                returning="empty_list",
            )
            return []
        
        ##Step purpose: Search for insights
        results = self._memory_search.search(query, n_results=max_insights)
        
        ##Step purpose: Extract content from results
        return [content for content, distance in results]
    
    ##Method purpose: Retrieve all insights for a user
    def get_all_insights(self, username: str) -> list[Insight]:
        """
        Retrieve all insights for a specific user.
        
        Args:
            username: Username to get insights for
            
        Returns:
            List of Insight objects
            
        Raises:
            ValueError: If username is invalid
        """
        ##Step purpose: Validate username before use
        safe_username = validate_username(username)
        
        logger.debug(
            "getting_all_insights",
            username=safe_username,
        )
        
        all_insights: list[Insight] = []
        user_dir = self._insights_root / safe_username
        
        ##Step purpose: Verify path is within base directory
        try:
            validate_path_within_base(user_dir, self._insights_root)
        except ValueError:
            ##Step purpose: Invalid path, return empty list
            logger.warning("invalid_username_path", username=username)
            return []
        
        ##Condition purpose: Return empty if user directory doesn't exist
        if not user_dir.exists():
            logger.debug(
                "no_insights_directory",
                username=username,
            )
            return []
        
        ##Loop purpose: Iterate through topic directories
        for topic_dir in user_dir.iterdir():
            ##Condition purpose: Skip non-directories
            if not topic_dir.is_dir():
                continue
            
            ##Loop purpose: Read insight files in topic directory
            for insight_file in topic_dir.glob("*.json"):
                ##Error purpose: Handle invalid insight files gracefully
                try:
                    with open(insight_file, encoding="utf-8") as f:
                        data = json.load(f)
                    
                    insight = Insight.from_dict(data)
                    all_insights.append(insight)
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(
                        "invalid_insight_file",
                        filepath=str(insight_file),
                        error=str(e),
                    )
                    continue
        
        logger.debug(
            "found_insights",
            username=username,
            count=len(all_insights),
        )
        
        return all_insights
    
    ##Method purpose: Get the most recently saved insight ID for a user
    def get_latest_insight_id(self, username: str) -> CortexId | None:
        """
        Get the cortex ID of the most recently saved insight.
        
        Args:
            username: Username to get latest insight for
            
        Returns:
            Cortex ID if found, None otherwise
        """
        all_insights = self.get_all_insights(username)
        
        ##Condition purpose: Return None if no insights
        if not all_insights:
            return None
        
        ##Step purpose: Sort by timestamp and get latest
        latest = max(
            all_insights,
            key=lambda i: i.timestamp,
        )
        
        return latest.cortex_id
    
    ##Method purpose: Get insights by topic
    def get_insights_by_topic(
        self,
        username: str,
        topic: str,
    ) -> list[Insight]:
        """
        Get all insights for a specific topic.
        
        Args:
            username: Username to filter by
            topic: Topic name to filter by
            
        Returns:
            List of insights for the topic
            
        Raises:
            ValueError: If username or topic is invalid
        """
        ##Step purpose: Validate username and topic before use
        safe_username = validate_username(username)
        safe_topic = validate_topic(topic)
        
        topic_dir = self._insights_root / safe_username / safe_topic
        
        ##Step purpose: Verify path is within base directory
        try:
            validate_path_within_base(topic_dir, self._insights_root)
        except ValueError:
            ##Step purpose: Invalid path, return empty list
            logger.warning("invalid_path", username=username, topic=topic)
            return []
        
        ##Condition purpose: Return empty if topic directory doesn't exist
        if not topic_dir.exists():
            return []
        
        insights: list[Insight] = []
        
        ##Loop purpose: Read insight files
        for insight_file in topic_dir.glob("*.json"):
            try:
                with open(insight_file, encoding="utf-8") as f:
                    data = json.load(f)
                insights.append(Insight.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue
        
        return insights
    
    ##Method purpose: Get all topics for a user
    def get_user_topics(self, username: str) -> list[str]:
        """
        Get all topics a user has insights under.
        
        Args:
            username: Username to get topics for
            
        Returns:
            List of topic names
            
        Raises:
            ValueError: If username is invalid
        """
        ##Step purpose: Validate username before use
        safe_username = validate_username(username)
        
        user_dir = self._insights_root / safe_username
        
        ##Step purpose: Verify path is within base directory
        try:
            validate_path_within_base(user_dir, self._insights_root)
        except ValueError:
            ##Step purpose: Invalid path, return empty list
            logger.warning("invalid_username_path", username=username)
            return []
        
        ##Condition purpose: Return empty if user directory doesn't exist
        if not user_dir.exists():
            return []
        
        return [
            d.name for d in user_dir.iterdir()
            if d.is_dir()
        ]
    
    ##Method purpose: Access the concern manager
    @property
    def concern_manager(self) -> ConcernManager:
        """Get the concern manager."""
        return self._concern_manager
    
    ##Method purpose: Factory method for production use
    @classmethod
    def create(
        cls,
        insights_root: Path,
        graph: GraphProtocol,
        llm: LLMProtocol,
        memory_search: MemorySearchProtocol | None = None,
    ) -> "InsightManager":
        """
        Factory method to create InsightManager.
        
        Args:
            insights_root: Root directory for insight storage
            graph: Cognitive graph for node linking
            llm: LLM interface for topic classification
            memory_search: Optional memory search for semantic retrieval
            
        Returns:
            Configured InsightManager instance
        """
        return cls(
            insights_root=insights_root,
            graph=graph,
            llm=llm,
            memory_search=memory_search,
        )
