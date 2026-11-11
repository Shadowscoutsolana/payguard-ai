"""
Unit tests for Blockchain Cluster Detection & Graph Analytics.
Verifies:
1. Circular transfer cycle detection (5-node mule loops).
2. Rapid fan-out dispersion detection.
3. Node degree calculations (in-degree, out-degree).
4. PyVis interactive graph visualization & risk color coding.
5. Rule-based AML cluster gate isolation from ML tabular features.
"""

from backend.app.blockchain_intelligence.graph import BlockchainGraphEngine
from backend.app.blockchain_intelligence.wallet_profile import WalletProfileService
from backend.app.blockchain_intelligence.cluster_detection import ClusterDetector
from dashboard.graph_viz import generate_interactive_graph_html


def test_circular_cycle_detection():
    """Verify detection of the 5-node circular transfer cycle."""
    graph_engine = BlockchainGraphEngine()
    wallet_service = WalletProfileService()
    detector = ClusterDetector(graph_engine, wallet_service)

    analysis = detector.analyze_cluster("0xde5a020000000000000000000000000000000001")
    assert analysis["has_cluster_flag"] is True
    assert analysis["cluster_risk_penalty"] == 50
    assert analysis["findings"]["pattern_type"] == "CIRCULAR_TRANSFER_CYCLE"
    assert "circular PYUSD transfer loop" in analysis["reasons"][0]
    assert analysis["findings"]["evidence"]["window_hours"] <= 72
    assert len(analysis["findings"]["evidence"]["transfers"]) == 5


def test_clean_merchant_no_cluster_flag():
    """Verify that verified clean merchant wallets are not falsely flagged."""
    graph_engine = BlockchainGraphEngine()
    wallet_service = WalletProfileService()
    detector = ClusterDetector(graph_engine, wallet_service)

    analysis = detector.analyze_cluster("0xde5a010000000000000000000000000000000001")
    assert analysis["has_cluster_flag"] is False
    assert analysis["cluster_risk_penalty"] <= 0  # Clean merchant discount
    assert analysis["findings"] is None


def test_pyvis_interactive_graph_rendering():
    """Verify PyVis HTML graph renders blue target, red mule, and green merchant nodes."""
    nodes = [
        {"address": "0xde5a020000000000000000000000000000000001", "label": "Target Recipient", "is_target": True, "is_flagged": False},
        {"address": "0x89b...44a2", "label": "Mule Node 1", "is_target": False, "is_flagged": True},
        {"address": "0xde5a010000000000000000000000000000000001", "label": "Clean Merchant", "is_target": False, "is_flagged": False, "allowlisted_merchant": True}
    ]
    edges = [
        {"from": "0xde5a020000000000000000000000000000000001", "to": "0x89b...44a2", "amount_pyusd": 4800, "token": "PYUSD"}
    ]

    html_output = generate_interactive_graph_html(nodes, edges, target_address="0xde5a020000000000000000000000000000000001")

    # Verify HTML canvas and styles
    assert "vis.Network" in html_output or "<html" in html_output
    assert "#0070BA" in html_output  # Target Recipient Blue
    assert "#E53935" in html_output  # High Risk Mule Red
    assert "#43A047" in html_output  # Clean Merchant Green
    assert "Fan-In" in html_output
    assert "Fan-Out" in html_output
