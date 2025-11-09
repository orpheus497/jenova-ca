# The JENOVA Cognitive Architecture - Graph Exporter
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 28: Graph Exporter - Export cortex knowledge graph to various formats.

Exports cortex graph to GraphML, DOT, JSON, and other formats for
visualization and analysis. All operations are 100% offline and FOSS.
"""

import json
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from datetime import datetime
import networkx as nx


class GraphExporter:
    """
    Export cortex knowledge graph to various formats.

    Supports GraphML, DOT (Graphviz), JSON, and custom HTML formats.
    All exports are self-contained and work offline.

    Example:
        >>> exporter = GraphExporter(cortex_graph)
        >>> exporter.to_graphml("graph.graphml")
        >>> exporter.to_json("graph.json")
        >>> exporter.to_html("graph.html")
    """

    def __init__(self, cortex_graph: Optional[nx.DiGraph] = None):
        """
        Initialize graph exporter.

        Args:
            cortex_graph: NetworkX directed graph from cortex
        """
        self.graph = cortex_graph if cortex_graph is not None else nx.DiGraph()

    def set_graph(self, cortex_graph: nx.DiGraph) -> None:
        """
        Set the graph to export.

        Args:
            cortex_graph: NetworkX directed graph
        """
        self.graph = cortex_graph

    def to_graphml(self, output_path: Path) -> None:
        """
        Export to GraphML format (XML-based graph format).

        GraphML is a standard format supported by many graph tools.

        Args:
            output_path: Output file path

        Example:
            >>> exporter.to_graphml(Path("cortex.graphml"))
        """
        output_path = Path(output_path)
        nx.write_graphml(self.graph, str(output_path))

    def to_dot(self, output_path: Path) -> None:
        """
        Export to DOT format (Graphviz).

        DOT can be rendered with Graphviz tools (100% FOSS).

        Args:
            output_path: Output file path

        Example:
            >>> exporter.to_dot(Path("cortex.dot"))
            # Then render with: dot -Tpng cortex.dot -o cortex.png
        """
        output_path = Path(output_path)

        # Build DOT format manually for better control
        dot_lines = ["digraph cortex {"]
        dot_lines.append('  rankdir=LR;')
        dot_lines.append('  node [shape=box, style=rounded];')

        # Add nodes
        for node_id, node_data in self.graph.nodes(data=True):
            label = node_data.get("label", str(node_id))
            node_type = node_data.get("type", "unknown")

            # Escape quotes in label
            label = label.replace('"', '\\"')

            # Color by type
            color_map = {
                "insight": "lightblue",
                "concept": "lightgreen",
                "memory": "lightyellow",
                "goal": "lightcoral",
            }
            color = color_map.get(node_type, "white")

            dot_lines.append(
                f'  "{node_id}" [label="{label}", fillcolor="{color}", style="filled,rounded"];'
            )

        # Add edges
        for source, target, edge_data in self.graph.edges(data=True):
            relationship = edge_data.get("relationship", "related")
            weight = edge_data.get("weight", 1.0)

            dot_lines.append(
                f'  "{source}" -> "{target}" [label="{relationship}", weight={weight}];'
            )

        dot_lines.append("}")

        output_path.write_text("\n".join(dot_lines), encoding="utf-8")

    def to_json(self, output_path: Path, pretty: bool = True) -> None:
        """
        Export to JSON format (node-link format).

        Args:
            output_path: Output file path
            pretty: Pretty-print JSON (default True)

        Example:
            >>> exporter.to_json(Path("cortex.json"))
        """
        output_path = Path(output_path)

        # Convert to node-link format
        data = nx.node_link_data(self.graph)

        # Add metadata
        data["metadata"] = {
            "exported_at": datetime.now().isoformat(),
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "graph_type": "cortex_knowledge_graph",
        }

        # Write JSON
        with open(output_path, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)

    def to_html(self, output_path: Path, title: str = "JENOVA Cortex Graph") -> None:
        """
        Export to self-contained HTML with embedded visualization.

        Uses D3.js-inspired force-directed layout (pure JavaScript, no CDN).
        Works 100% offline.

        Args:
            output_path: Output file path
            title: HTML page title

        Example:
            >>> exporter.to_html(Path("cortex.html"))
        """
        output_path = Path(output_path)

        # Convert graph to JSON for JavaScript
        graph_data = nx.node_link_data(self.graph)

        # Build self-contained HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1e1e1e;
            color: #e0e0e0;
            overflow: hidden;
        }}
        #graph-container {{
            width: 100vw;
            height: 100vh;
            position: relative;
        }}
        canvas {{
            display: block;
        }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(30, 30, 30, 0.9);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #444;
            font-size: 12px;
            max-width: 300px;
        }}
        #controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(30, 30, 30, 0.9);
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #444;
        }}
        button {{
            background: #2d2d30;
            border: 1px solid #555;
            color: #e0e0e0;
            padding: 8px 12px;
            margin: 5px;
            border-radius: 4px;
            cursor: pointer;
        }}
        button:hover {{
            background: #3e3e42;
        }}
        #node-details {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(30, 30, 30, 0.9);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #444;
            max-width: 400px;
            max-height: 200px;
            overflow-y: auto;
            display: none;
        }}
    </style>
</head>
<body>
    <div id="graph-container">
        <canvas id="graph-canvas"></canvas>
        <div id="info">
            <h3 style="margin: 0 0 10px 0;">{title}</h3>
            <div id="stats"></div>
        </div>
        <div id="controls">
            <button onclick="resetZoom()">Reset View</button>
            <button onclick="togglePhysics()">Toggle Physics</button>
        </div>
        <div id="node-details"></div>
    </div>

    <script>
        const graphData = {json.dumps(graph_data)};

        const canvas = document.getElementById('graph-canvas');
        const ctx = canvas.getContext('2d');
        const statsDiv = document.getElementById('stats');
        const detailsDiv = document.getElementById('node-details');

        // Set canvas size
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        window.addEventListener('resize', () => {{
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            draw();
        }});

        // Physics simulation
        let nodes = [];
        let edges = [];
        let physicsEnabled = true;
        let zoom = 1;
        let panX = 0;
        let panY = 0;
        let isDragging = false;
        let dragNode = null;
        let selectedNode = null;

        // Initialize nodes and edges
        function initGraph() {{
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;

            nodes = graphData.nodes.map((node, i) => ({{
                id: node.id,
                label: node.label || node.id,
                type: node.type || 'unknown',
                x: centerX + Math.cos(i * 2 * Math.PI / graphData.nodes.length) * 200,
                y: centerY + Math.sin(i * 2 * Math.PI / graphData.nodes.length) * 200,
                vx: 0,
                vy: 0,
                data: node
            }}));

            edges = graphData.links.map(link => ({{
                source: nodes.find(n => n.id === link.source),
                target: nodes.find(n => n.id === link.target),
                relationship: link.relationship || 'related',
                weight: link.weight || 1
            }}));

            updateStats();
        }}

        function updateStats() {{
            statsDiv.innerHTML = `
                <strong>Nodes:</strong> ${{nodes.length}}<br>
                <strong>Edges:</strong> ${{edges.length}}<br>
                <strong>Physics:</strong> ${{physicsEnabled ? 'On' : 'Off'}}
            `;
        }}

        // Force-directed layout
        function applyForces() {{
            if (!physicsEnabled) return;

            const repulsion = 5000;
            const attraction = 0.01;
            const damping = 0.85;

            // Repulsion between all nodes
            for (let i = 0; i < nodes.length; i++) {{
                for (let j = i + 1; j < nodes.length; j++) {{
                    const dx = nodes[j].x - nodes[i].x;
                    const dy = nodes[j].y - nodes[i].y;
                    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                    const force = repulsion / (dist * dist);
                    const fx = (dx / dist) * force;
                    const fy = (dy / dist) * force;

                    nodes[i].vx -= fx;
                    nodes[i].vy -= fy;
                    nodes[j].vx += fx;
                    nodes[j].vy += fy;
                }}
            }}

            // Attraction along edges
            edges.forEach(edge => {{
                const dx = edge.target.x - edge.source.x;
                const dy = edge.target.y - edge.source.y;
                const force = attraction * edge.weight;

                edge.source.vx += dx * force;
                edge.source.vy += dy * force;
                edge.target.vx -= dx * force;
                edge.target.vy -= dy * force;
            }});

            // Apply velocities with damping
            nodes.forEach(node => {{
                if (node !== dragNode) {{
                    node.x += node.vx;
                    node.y += node.vy;
                    node.vx *= damping;
                    node.vy *= damping;
                }}
            }});
        }}

        function draw() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.save();

            // Apply zoom and pan
            ctx.translate(panX, panY);
            ctx.scale(zoom, zoom);

            // Draw edges
            ctx.strokeStyle = '#555';
            ctx.lineWidth = 1;
            edges.forEach(edge => {{
                ctx.beginPath();
                ctx.moveTo(edge.source.x, edge.source.y);
                ctx.lineTo(edge.target.x, edge.target.y);
                ctx.stroke();
            }});

            // Draw nodes
            nodes.forEach(node => {{
                const colors = {{
                    insight: '#4fc3f7',
                    concept: '#81c784',
                    memory: '#fff176',
                    goal: '#e57373',
                    unknown: '#bdbdbd'
                }};

                ctx.fillStyle = colors[node.type] || colors.unknown;
                ctx.strokeStyle = node === selectedNode ? '#fff' : '#333';
                ctx.lineWidth = node === selectedNode ? 3 : 1;

                ctx.beginPath();
                ctx.arc(node.x, node.y, 8, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();

                // Draw label
                if (zoom > 0.5) {{
                    ctx.fillStyle = '#e0e0e0';
                    ctx.font = '12px sans-serif';
                    ctx.fillText(node.label.substring(0, 20), node.x + 12, node.y + 4);
                }}
            }});

            ctx.restore();
        }}

        function animate() {{
            applyForces();
            draw();
            requestAnimationFrame(animate);
        }}

        // Mouse interaction
        let mouseDown = false;
        let lastX = 0;
        let lastY = 0;

        canvas.addEventListener('mousedown', (e) => {{
            mouseDown = true;
            lastX = e.clientX;
            lastY = e.clientY;

            // Check if clicking on a node
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left - panX) / zoom;
            const y = (e.clientY - rect.top - panY) / zoom;

            dragNode = nodes.find(n => {{
                const dx = n.x - x;
                const dy = n.y - y;
                return Math.sqrt(dx * dx + dy * dy) < 8;
            }});

            if (dragNode) {{
                selectedNode = dragNode;
                showNodeDetails(selectedNode);
            }}
        }});

        canvas.addEventListener('mousemove', (e) => {{
            if (!mouseDown) return;

            if (dragNode) {{
                const rect = canvas.getBoundingClientRect();
                dragNode.x = (e.clientX - rect.left - panX) / zoom;
                dragNode.y = (e.clientY - rect.top - panY) / zoom;
                dragNode.vx = 0;
                dragNode.vy = 0;
            }} else {{
                panX += e.clientX - lastX;
                panY += e.clientY - lastY;
            }}

            lastX = e.clientX;
            lastY = e.clientY;
        }});

        canvas.addEventListener('mouseup', () => {{
            mouseDown = false;
            dragNode = null;
        }});

        canvas.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            zoom = Math.max(0.1, Math.min(3, zoom * delta));
        }});

        function showNodeDetails(node) {{
            detailsDiv.style.display = 'block';
            detailsDiv.innerHTML = `
                <h4 style="margin: 0 0 10px 0;">${{node.label}}</h4>
                <strong>Type:</strong> ${{node.type}}<br>
                <strong>ID:</strong> ${{node.id}}<br>
            `;
        }}

        function resetZoom() {{
            zoom = 1;
            panX = 0;
            panY = 0;
        }}

        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            updateStats();
        }}

        // Initialize and start
        initGraph();
        animate();
    </script>
</body>
</html>"""

        output_path.write_text(html, encoding="utf-8")

    def to_gexf(self, output_path: Path) -> None:
        """
        Export to GEXF format (Gephi-compatible).

        Args:
            output_path: Output file path

        Example:
            >>> exporter.to_gexf(Path("cortex.gexf"))
        """
        output_path = Path(output_path)
        nx.write_gexf(self.graph, str(output_path))

    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get graph statistics.

        Returns:
            Dict with graph metrics

        Example:
            >>> stats = exporter.get_graph_stats()
            >>> print(stats["node_count"], stats["edge_count"])
        """
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "is_directed": self.graph.is_directed(),
            "is_connected": nx.is_weakly_connected(self.graph) if self.graph.is_directed() else nx.is_connected(self.graph),
        }

    def export_subgraph(
        self,
        node_ids: List[str],
        output_path: Path,
        format: str = "json"
    ) -> None:
        """
        Export a subgraph containing specific nodes.

        Args:
            node_ids: List of node IDs to include
            output_path: Output file path
            format: Export format (json, graphml, dot, html)

        Example:
            >>> exporter.export_subgraph(
            ...     ["node1", "node2"],
            ...     Path("sub.json"),
            ...     format="json"
            ... )
        """
        # Create subgraph
        subgraph = self.graph.subgraph(node_ids).copy()

        # Create temporary exporter with subgraph
        temp_exporter = GraphExporter(subgraph)

        # Export based on format
        if format == "json":
            temp_exporter.to_json(output_path)
        elif format == "graphml":
            temp_exporter.to_graphml(output_path)
        elif format == "dot":
            temp_exporter.to_dot(output_path)
        elif format == "html":
            temp_exporter.to_html(output_path)
        elif format == "gexf":
            temp_exporter.to_gexf(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
