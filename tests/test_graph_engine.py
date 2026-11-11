"""
Unit tests for the NetworkX PYUSD transfer graph.
"""

from backend.app.blockchain_intelligence.graph import BlockchainGraphEngine
from backend.app.blockchain_intelligence.wallet_profile import WalletProfileService

RING = "0xde5a020000000000000000000000000000000001"
MERCHANT = "0xde5a010000000000000000000000000000000001"


def test_cycle_is_found_from_transfers():
    graph_engine = BlockchainGraphEngine()
    cycles = graph_engine.cycles_through(RING, max_length=6)
    assert len(cycles) == 1 and len(cycles[0]) == 5
    transfers = graph_engine.cycle_transfers(cycles[0])
    assert [t["transfer_id"] for t in transfers] == [f"SYN-RING-00{i}" for i in range(1, 6)]
    metrics = graph_engine.get_node_metrics(RING)
    assert metrics["in_degree"] == 1 and metrics["out_degree"] == 1


def test_no_cycle_longer_than_bound():
    assert BlockchainGraphEngine().cycles_through(RING, max_length=4) == []


def test_clean_merchant_profile():
    profile = WalletProfileService().get_wallet_profile(MERCHANT)
    assert profile["intelligence_status"] == "KNOWN"
    assert profile["allowlisted_merchant"] is True
    assert profile["provenance"] == "synthetic"


def test_lookup_is_case_insensitive_and_exact():
    service = WalletProfileService()
    assert service.get_wallet_profile(MERCHANT.upper().replace("0X", "0x"))["intelligence_status"] == "KNOWN"
    assert service.get_wallet_profile(MERCHANT[:-1])["intelligence_status"] == "UNKNOWN"


def test_all_demo_addresses_are_full_length_and_synthetic():
    data = BlockchainGraphEngine().data
    demo_wallets = {a: w for a, w in data["wallets"].items() if w.get("cluster_id")}
    assert demo_wallets
    for addr, info in demo_wallets.items():
        assert len(addr) == 42 and addr.startswith("0xde5a"), addr
        assert info["provenance"] == "synthetic"
    for t in (t for t in data["transfers"] if t.get("cluster_id")):
        assert t["tx_hash"] is None and t["transfer_id"].startswith("SYN-")


def test_real_data_if_present_is_never_mixed_with_synthetic_labels():
    data = BlockchainGraphEngine().data
    for addr, info in data["wallets"].items():
        if info["provenance"] == "real_public_chain":
            assert not addr.startswith("0xde5a") and info.get("cluster_id") is None
    for t in data["transfers"]:
        if t["provenance"] == "real_public_chain":
            assert t["tx_hash"] and len(t["tx_hash"]) == 66
