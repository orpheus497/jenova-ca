##Script function and purpose: Insights package for The JENOVA Cognitive Architecture
"""
Insights Package

This package manages insight generation, organization by concerns,
and meta-insight synthesis. Insights are stored persistently and
linked through the Cortex knowledge graph.
"""

from jenova.insights.types import (
    Concern,
    CortexId,
    Insight,
    InsightSearchResult,
)
from jenova.insights.concerns import ConcernManager
from jenova.insights.manager import InsightManager

__all__ = [
    "Concern",
    "ConcernManager",
    "CortexId",
    "Insight",
    "InsightManager",
    "InsightSearchResult",
]
