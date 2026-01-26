##Script function and purpose: Assumptions package for The JENOVA Cognitive Architecture
"""
Assumptions Package

This package manages assumption tracking, validation, and lifecycle.
Assumptions are hypotheses that JENOVA forms about the user and world,
which can be verified or disproven through conversation.
"""

from jenova.assumptions.manager import AssumptionManager
from jenova.assumptions.types import (
    Assumption,
    AssumptionStatus,
    AssumptionStore,
    CortexId,
)

__all__ = [
    "Assumption",
    "AssumptionManager",
    "AssumptionStatus",
    "AssumptionStore",
    "CortexId",
]
