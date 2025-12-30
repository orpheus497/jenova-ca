##Script function and purpose: Cognitive Graph Data Structures for The JENOVA Cognitive Architecture
##This module defines dataclass structures for nodes and links in the cognitive graph

from dataclasses import dataclass, field
from datetime import datetime
import uuid

##Class purpose: Represents a single node in the cognitive graph with metadata
@dataclass
class CognitiveNode:
    """Represents a single node in the cognitive graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: str # e.g., 'insight', 'memory', 'assumption', 'meta-insight'
    content: str
    user: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)

##Class purpose: Represents a directed link between two nodes with relationship type
@dataclass
class CognitiveLink:
    """Represents a directed link between two nodes in the cognitive graph."""
    source_id: str
    target_id: str
    relationship: str # e.g., 'elaborates_on', 'conflicts_with', 'created_from'
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
