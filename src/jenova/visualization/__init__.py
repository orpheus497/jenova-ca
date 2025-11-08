# The JENOVA Cognitive Architecture - Visualization Module
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 28: Knowledge Graph Visualization - Offline graph visualization and analysis.

Provides graph export, terminal rendering, and analysis tools for the
cortex knowledge graph. All components are 100% offline and FOSS.

Components:
    - GraphExporter: Export to GraphML, DOT, JSON, HTML formats
    - TerminalRenderer: ASCII art visualization for terminal
    - GraphAnalyzer: Centrality, clustering, community detection

Example:
    >>> from jenova.visualization import GraphExporter, TerminalRenderer, GraphAnalyzer
    >>> from pathlib import Path
    >>>
    >>> # Export graph
    >>> exporter = GraphExporter(cortex_graph)
    >>> exporter.to_html(Path("graph.html"))
    >>> exporter.to_json(Path("graph.json"))
    >>>
    >>> # Render in terminal
    >>> renderer = TerminalRenderer(cortex_graph)
    >>> print(renderer.render_tree("root_node"))
    >>> print(renderer.render_stats())
    >>>
    >>> # Analyze graph
    >>> analyzer = GraphAnalyzer(cortex_graph)
    >>> important = analyzer.find_most_important_nodes("pagerank", top_n=10)
    >>> communities = analyzer.detect_communities()
"""

from jenova.visualization.graph_exporter import GraphExporter
from jenova.visualization.terminal_renderer import TerminalRenderer
from jenova.visualization.graph_analyzer import GraphAnalyzer

__all__ = [
    "GraphExporter",
    "TerminalRenderer",
    "GraphAnalyzer",
]
