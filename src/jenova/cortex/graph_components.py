
# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CognitiveNode:
    """Represents a single node in the cognitive graph."""
    node_type: str  # e.g., 'insight', 'memory', 'assumption', 'meta-insight'
    content: str
    user: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)


@dataclass
class CognitiveLink:
    """Represents a directed link between two nodes in the cognitive graph."""
    source_id: str
    target_id: str
    relationship: str  # e.g., 'elaborates_on', 'conflicts_with', 'created_from'
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
