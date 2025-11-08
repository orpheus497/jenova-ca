# The JENOVA Cognitive Architecture - Terminal Renderer
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 28: Terminal Renderer - ASCII art visualization for terminal.

Renders knowledge graph as ASCII art tree or network diagram.
Works in any terminal, 100% offline, no external dependencies.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import networkx as nx


class TerminalRenderer:
    """
    Render graph as ASCII art for terminal display.

    Supports tree view, list view, and simple network diagram.
    All rendering is done with pure ASCII characters.

    Example:
        >>> renderer = TerminalRenderer(graph)
        >>> print(renderer.render_tree("root_node"))
        >>> print(renderer.render_list())
        >>> print(renderer.render_network())
    """

    def __init__(self, graph: Optional[nx.DiGraph] = None):
        """
        Initialize terminal renderer.

        Args:
            graph: NetworkX directed graph
        """
        self.graph = graph if graph is not None else nx.DiGraph()

    def set_graph(self, graph: nx.DiGraph) -> None:
        """
        Set the graph to render.

        Args:
            graph: NetworkX directed graph
        """
        self.graph = graph

    def render_tree(
        self,
        root_node: str,
        max_depth: int = 3,
        show_types: bool = True
    ) -> str:
        """
        Render graph as ASCII tree from root node.

        Args:
            root_node: Starting node ID
            max_depth: Maximum depth to traverse
            show_types: Show node types in parentheses

        Returns:
            ASCII tree representation

        Example:
            >>> tree = renderer.render_tree("insight_42", max_depth=2)
            >>> print(tree)
        """
        if root_node not in self.graph:
            return f"Node '{root_node}' not found in graph"

        lines = []
        visited = set()

        def render_node(node_id: str, prefix: str, depth: int):
            if depth > max_depth or node_id in visited:
                return

            visited.add(node_id)

            # Get node data
            node_data = self.graph.nodes[node_id]
            label = node_data.get("label", node_id)
            node_type = node_data.get("type", "unknown")

            # Format node line
            type_str = f" ({node_type})" if show_types else ""
            node_line = f"{label}{type_str}"

            lines.append(f"{prefix}{node_line}")

            # Get children (outgoing edges)
            children = list(self.graph.successors(node_id))

            if children:
                for i, child in enumerate(children):
                    is_last = (i == len(children) - 1)

                    if is_last:
                        child_prefix = prefix + "└── "
                        extension = prefix + "    "
                    else:
                        child_prefix = prefix + "├── "
                        extension = prefix + "│   "

                    # Get edge data
                    edge_data = self.graph[node_id][child]
                    relationship = edge_data.get("relationship", "")

                    if relationship:
                        lines.append(f"{child_prefix}[{relationship}]")
                        render_node(child, extension + "    ", depth + 1)
                    else:
                        render_node(child, child_prefix, depth + 1)

        render_node(root_node, "", 0)
        return "\n".join(lines)

    def render_list(
        self,
        group_by_type: bool = True,
        show_edges: bool = True
    ) -> str:
        """
        Render graph as formatted list.

        Args:
            group_by_type: Group nodes by type
            show_edges: Show outgoing edges for each node

        Returns:
            Formatted list representation

        Example:
            >>> list_view = renderer.render_list(group_by_type=True)
            >>> print(list_view)
        """
        lines = []

        if group_by_type:
            # Group nodes by type
            by_type: Dict[str, List[str]] = {}
            for node_id, node_data in self.graph.nodes(data=True):
                node_type = node_data.get("type", "unknown")
                if node_type not in by_type:
                    by_type[node_type] = []
                by_type[node_type].append(node_id)

            # Render each type group
            for node_type, node_ids in sorted(by_type.items()):
                lines.append(f"\n[{node_type.upper()}] ({len(node_ids)} nodes)")
                lines.append("=" * 50)

                for node_id in sorted(node_ids):
                    node_data = self.graph.nodes[node_id]
                    label = node_data.get("label", node_id)
                    lines.append(f"  • {label}")

                    if show_edges:
                        edges = list(self.graph.out_edges(node_id, data=True))
                        if edges:
                            for _, target, edge_data in edges:
                                relationship = edge_data.get("relationship", "→")
                                target_label = self.graph.nodes[target].get("label", target)
                                lines.append(f"      {relationship} {target_label}")
        else:
            # Simple list
            lines.append(f"NODES ({self.graph.number_of_nodes()})")
            lines.append("=" * 50)

            for node_id, node_data in self.graph.nodes(data=True):
                label = node_data.get("label", node_id)
                node_type = node_data.get("type", "unknown")
                lines.append(f"  [{node_type}] {label}")

                if show_edges:
                    edges = list(self.graph.out_edges(node_id, data=True))
                    if edges:
                        for _, target, edge_data in edges:
                            relationship = edge_data.get("relationship", "→")
                            target_label = self.graph.nodes[target].get("label", target)
                            lines.append(f"      {relationship} {target_label}")

        return "\n".join(lines)

    def render_network(
        self,
        width: int = 80,
        height: int = 24,
        max_nodes: int = 50
    ) -> str:
        """
        Render simple network diagram.

        Creates a grid-based visualization with nodes and connections.
        Limited to max_nodes for readability.

        Args:
            width: Display width in characters
            height: Display height in characters
            max_nodes: Maximum nodes to display

        Returns:
            ASCII network diagram

        Example:
            >>> diagram = renderer.render_network(width=80, height=20)
            >>> print(diagram)
        """
        if self.graph.number_of_nodes() == 0:
            return "Empty graph"

        # Limit nodes
        node_list = list(self.graph.nodes())[:max_nodes]

        if len(node_list) == 0:
            return "No nodes to display"

        # Simple grid layout
        grid_size = int(len(node_list) ** 0.5) + 1
        positions = {}

        for i, node_id in enumerate(node_list):
            row = i // grid_size
            col = i % grid_size

            x = int((col + 1) * width / (grid_size + 1))
            y = int((row + 1) * height / (grid_size + 1))

            positions[node_id] = (x, y)

        # Create canvas
        canvas = [[' ' for _ in range(width)] for _ in range(height)]

        # Draw edges first (background)
        for node_id in node_list:
            if node_id not in positions:
                continue

            x1, y1 = positions[node_id]

            for target in self.graph.successors(node_id):
                if target not in positions:
                    continue

                x2, y2 = positions[target]

                # Draw simple line
                self._draw_line(canvas, x1, y1, x2, y2, '.')

        # Draw nodes (foreground)
        for node_id in node_list:
            if node_id not in positions:
                continue

            x, y = positions[node_id]

            # Get node label (first char)
            node_data = self.graph.nodes[node_id]
            node_type = node_data.get("type", "unknown")

            # Type symbols
            symbols = {
                "insight": "I",
                "concept": "C",
                "memory": "M",
                "goal": "G",
                "unknown": "•"
            }
            symbol = symbols.get(node_type, "•")

            if 0 <= y < height and 0 <= x < width:
                canvas[y][x] = symbol

        # Convert canvas to string
        lines = [''.join(row) for row in canvas]

        # Add legend
        legend = "\nLegend: I=Insight C=Concept M=Memory G=Goal •=Unknown"
        lines.append(legend)

        # Add stats
        stats = f"Showing {len(node_list)} of {self.graph.number_of_nodes()} nodes"
        lines.append(stats)

        return '\n'.join(lines)

    def _draw_line(
        self,
        canvas: List[List[str]],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        char: str
    ) -> None:
        """
        Draw line on canvas using Bresenham's algorithm.

        Args:
            canvas: 2D character canvas
            x1, y1: Start position
            x2, y2: End position
            char: Character to draw
        """
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        x, y = x1, y1

        while True:
            # Draw point if in bounds and not overwriting a node
            if (0 <= y < len(canvas) and 0 <= x < len(canvas[0]) and
                canvas[y][x] == ' '):
                canvas[y][x] = char

            if x == x2 and y == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def render_stats(self) -> str:
        """
        Render graph statistics as formatted text.

        Returns:
            Formatted statistics

        Example:
            >>> stats = renderer.render_stats()
            >>> print(stats)
        """
        lines = []
        lines.append("GRAPH STATISTICS")
        lines.append("=" * 50)

        # Basic stats
        lines.append(f"Nodes: {self.graph.number_of_nodes()}")
        lines.append(f"Edges: {self.graph.number_of_edges()}")

        # Density
        if self.graph.number_of_nodes() > 1:
            density = nx.density(self.graph)
            lines.append(f"Density: {density:.3f}")

        # Node types
        types: Dict[str, int] = {}
        for _, node_data in self.graph.nodes(data=True):
            node_type = node_data.get("type", "unknown")
            types[node_type] = types.get(node_type, 0) + 1

        if types:
            lines.append("\nNode Types:")
            for node_type, count in sorted(types.items()):
                percentage = (count / self.graph.number_of_nodes()) * 100
                lines.append(f"  {node_type}: {count} ({percentage:.1f}%)")

        # Relationship types
        relationships: Dict[str, int] = {}
        for _, _, edge_data in self.graph.edges(data=True):
            rel = edge_data.get("relationship", "unknown")
            relationships[rel] = relationships.get(rel, 0) + 1

        if relationships:
            lines.append("\nRelationship Types:")
            for rel, count in sorted(relationships.items()):
                percentage = (count / self.graph.number_of_edges()) * 100 if self.graph.number_of_edges() > 0 else 0
                lines.append(f"  {rel}: {count} ({percentage:.1f}%)")

        # Degree statistics
        if self.graph.number_of_nodes() > 0:
            in_degrees = [d for _, d in self.graph.in_degree()]
            out_degrees = [d for _, d in self.graph.out_degree()]

            lines.append("\nDegree Statistics:")
            lines.append(f"  Avg in-degree: {sum(in_degrees) / len(in_degrees):.2f}")
            lines.append(f"  Avg out-degree: {sum(out_degrees) / len(out_degrees):.2f}")
            lines.append(f"  Max in-degree: {max(in_degrees) if in_degrees else 0}")
            lines.append(f"  Max out-degree: {max(out_degrees) if out_degrees else 0}")

        return "\n".join(lines)

    def render_path(self, start: str, end: str) -> str:
        """
        Render shortest path between two nodes.

        Args:
            start: Start node ID
            end: End node ID

        Returns:
            Formatted path description

        Example:
            >>> path = renderer.render_path("node1", "node5")
            >>> print(path)
        """
        if start not in self.graph or end not in self.graph:
            return "One or both nodes not found in graph"

        try:
            path = nx.shortest_path(self.graph, start, end)
        except nx.NetworkXNoPath:
            return f"No path found from '{start}' to '{end}'"

        lines = []
        lines.append(f"Path from '{start}' to '{end}':")
        lines.append("=" * 50)

        for i in range(len(path)):
            node_id = path[i]
            node_data = self.graph.nodes[node_id]
            label = node_data.get("label", node_id)
            node_type = node_data.get("type", "unknown")

            # Format node
            lines.append(f"{i + 1}. [{node_type}] {label}")

            # Show edge to next node
            if i < len(path) - 1:
                next_node = path[i + 1]
                edge_data = self.graph[node_id][next_node]
                relationship = edge_data.get("relationship", "→")
                lines.append(f"     ↓ {relationship}")

        lines.append(f"\nPath length: {len(path) - 1} edges")

        return "\n".join(lines)
