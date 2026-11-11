"""
Interactive Blockchain Graph Visualizer using PyVis and NetworkX.
Renders an interactive, physics-enabled directed graph for Streamlit with:
- Blue node: current payment recipient
- Red nodes: wallets involved in a COMPUTED suspicious-pattern indicator
- Green nodes: allowlisted merchant (demo allowlist)
- Grey nodes: no indicator found (not proof of being clean)
All labels are HTML-escaped before rendering.
- Tooltips: Showing in-degree, out-degree, fan-in/fan-out, and transfer amounts.
"""

import html

from pyvis.network import Network
import networkx as nx
from typing import Dict, Any, List


def generate_interactive_graph_html(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    target_address: str,
    height: str = "450px",
    width: str = "100%"
) -> str:
    """
    Builds a PyVis network graph and returns its self-contained HTML representation.
    """
    net = Network(height=height, width=width, directed=True, bgcolor="#0e1117", font_color="#fafafa")

    # Physics configuration for smooth cluster separation
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springStrength": 0.08,
          "damping": 0.4
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35
      },
      "edges": {
        "smooth": {
          "type": "curvedCW",
          "roundness": 0.2
        },
        "arrows": {
          "to": { "enabled": true, "scaleFactor": 0.8 }
        },
        "color": { "color": "#888888", "highlight": "#00d2ff" }
      }
    }
    """)

    # Compute in/out degree for tooltip display
    in_degrees = {}
    out_degrees = {}
    for edge in edges:
        u = edge.get("from")
        v = edge.get("to")
        out_degrees[u] = out_degrees.get(u, 0) + 1
        in_degrees[v] = in_degrees.get(v, 0) + 1

    for node in nodes:
        addr = node.get("address") or node.get("id")
        is_target = (addr == target_address) or node.get("is_target", False)
        is_flagged = node.get("is_flagged", False)          # involved in a COMPUTED indicator's evidence
        is_allowlisted = node.get("allowlisted_merchant", False)
        label = html.escape(str(node.get("label", addr)))
        provenance = html.escape(str(node.get("provenance", "unknown")))

        in_deg = in_degrees.get(addr, 0)
        out_deg = out_degrees.get(addr, 0)

        # Blue = payment recipient · Red = in a computed suspicious-pattern indicator
        # Green = allowlisted merchant · Grey = no indicator found (NOT the same as "clean")
        if is_target:
            color, shape, size, border_color = "#0070BA", "diamond", 35, "#00d2ff"
            state = "Payment recipient" + (" (in suspicious pattern)" if is_flagged else "")
        elif is_flagged:
            color, shape, size, border_color = "#E53935", "dot", 26, "#FF8A80"
            state = "In a computed suspicious-pattern indicator"
        elif is_allowlisted:
            color, shape, size, border_color = "#43A047", "dot", 24, "#A5D6A7"
            state = "Allowlisted merchant (demo allowlist)"
        else:
            color, shape, size, border_color = "#9E9E9E", "dot", 22, "#E0E0E0"
            state = "No indicator found (not proof of being clean)"

        tooltip = (f"<b>Address:</b> {html.escape(addr)}<br/><b>Role:</b> {label}<br/>"
                   f"<b>State:</b> {state}<br/><b>Data:</b> {provenance}<br/>"
                   f"<b>Fan-In (In-Degree):</b> {in_deg}<br/><b>Fan-Out (Out-Degree):</b> {out_deg}")

        display_label = f"🎯 {addr[:8]}…{addr[-4:]}" if is_target else f"{addr[:6]}…{addr[-4:]}"

        net.add_node(
            addr,
            label=display_label,
            title=tooltip,
            color={"background": color, "border": border_color, "highlight": {"background": "#FFEB3B", "border": "#F57F17"}},
            shape=shape,
            size=size,
            font={"color": "#FFFFFF", "size": 13, "face": "Helvetica"}
        )

    for edge in edges:
        u = edge.get("from")
        v = edge.get("to")
        amount = edge.get("amount_pyusd") or edge.get("amount", 0)
        token = edge.get("token", "PYUSD")

        edge_title = f"Transfer: {amount:,.2f} {token}"
        edge_label = f"${amount:,.0f} {token}" if amount else token

        net.add_edge(
            u,
            v,
            title=edge_title,
            label=edge_label,
            color="#9E9E9E",
            width=2
        )

    return net.generate_html()
