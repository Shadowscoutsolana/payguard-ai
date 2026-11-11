"""
Layer 3: Blockchain Intelligence - NetworkX transfer graph.

Nodes are wallet addresses (lower-case), edges are observed PYUSD transfers.
Every edge keeps the exact transfers behind it (id, amount, timestamp, tx hash, provenance),
so any finding can cite its evidence.

Data sources:
  data/demo_wallets.json          synthetic demo wallets/transfers (provenance "synthetic")
  data/real_pyusd_transfers.json  optional, written by scripts/fetch_pyusd_transfers.py
                                  (provenance "real_public_chain", real tx hashes)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
DATA_PATH = DATA_DIR / "demo_wallets.json"
REAL_TRANSFERS_PATH = DATA_DIR / "real_pyusd_transfers.json"


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_blockchain_data(demo_path: Path = DATA_PATH, real_path: Path = REAL_TRANSFERS_PATH) -> Dict[str, Any]:
    """Returns {"wallets": {addr: info}, "transfers": [...], "clusters": {cluster_id: meta}}."""
    wallets: Dict[str, Dict[str, Any]] = {}
    transfers: List[Dict[str, Any]] = []
    clusters: Dict[str, Dict[str, Any]] = {}

    if demo_path.exists():
        data = json.loads(demo_path.read_text(encoding="utf-8"))
        for key, c in data.get("clusters", {}).items():
            cid = c["cluster_id"]
            members = [w["address"].lower() for w in c.get("wallets", [])]
            clusters[cid] = {"cluster_id": cid, "key": key, "description": c.get("description", ""),
                             "target_wallet": c.get("target_wallet", "").lower(), "members": members,
                             "provenance": "synthetic"}
            for w in c.get("wallets", []):
                wallets[w["address"].lower()] = {**w, "address": w["address"].lower(), "cluster_id": cid}
            for e in c.get("edges", []):
                transfers.append({**e, "from": e["from"].lower(), "to": e["to"].lower(), "cluster_id": cid})

    if real_path.exists():
        real = json.loads(real_path.read_text(encoding="utf-8"))
        for e in real.get("transfers", []):
            t = {**e, "from": e["from"].lower(), "to": e["to"].lower(), "provenance": "real_public_chain",
                 "cluster_id": None}
            transfers.append(t)
            for a in (t["from"], t["to"]):
                wallets.setdefault(a, {"address": a, "label": "Real PYUSD address (public chain)",
                                       "provenance": "real_public_chain", "cluster_id": None})

    return {"wallets": wallets, "transfers": transfers, "clusters": clusters}


class BlockchainGraphEngine:
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self.data = data or load_blockchain_data()
        self.graph = nx.DiGraph()
        for addr, info in self.data["wallets"].items():
            self.graph.add_node(addr, **{k: v for k, v in info.items() if k != "address"})
        for t in self.data["transfers"]:
            if not self.graph.has_edge(t["from"], t["to"]):
                self.graph.add_edge(t["from"], t["to"], transfers=[])
            self.graph[t["from"]][t["to"]]["transfers"].append(t)

    def has_wallet(self, address: Optional[str]) -> bool:
        return bool(address) and self.graph.has_node(address.lower())

    def get_node_metrics(self, address: str) -> Dict[str, Any]:
        a = (address or "").lower()
        if not self.graph.has_node(a):
            return {"in_degree": 0, "out_degree": 0, "neighbors": []}
        neighbors = set(self.graph.successors(a)) | set(self.graph.predecessors(a))
        return {"in_degree": self.graph.in_degree(a), "out_degree": self.graph.out_degree(a),
                "neighbors": sorted(neighbors)}

    def cycles_through(self, address: str, max_length: int = 6) -> List[List[str]]:
        """Simple directed cycles of at most max_length wallets that include `address`."""
        a = (address or "").lower()
        if not self.graph.has_node(a):
            return []
        return [c for c in nx.simple_cycles(self.graph, length_bound=max_length) if a in c]

    def cycle_transfers(self, cycle: List[str]) -> List[Dict[str, Any]]:
        """The transfers along a cycle (earliest transfer per hop), in chronological order."""
        hops = zip(cycle, cycle[1:] + cycle[:1])
        picked = [min(self.graph[u][v]["transfers"], key=lambda t: t["timestamp"]) for u, v in hops]
        return sorted(picked, key=lambda t: t["timestamp"])

    def out_transfers(self, address: str) -> List[Dict[str, Any]]:
        a = (address or "").lower()
        if not self.graph.has_node(a):
            return []
        return sorted((t for _, v in self.graph.out_edges(a) for t in self.graph[a][v]["transfers"]),
                      key=lambda t: t["timestamp"])

    def in_transfers(self, address: str) -> List[Dict[str, Any]]:
        a = (address or "").lower()
        if not self.graph.has_node(a):
            return []
        return sorted((t for u, _ in self.graph.in_edges(a) for t in self.graph[u][a]["transfers"]),
                      key=lambda t: t["timestamp"])

    def export_subgraph(self, target_address: str, depth: int = 2) -> Dict[str, Any]:
        """Nodes and edges around a wallet, for visualisation."""
        target = (target_address or "").lower()
        if not self.graph.has_node(target):
            return {"nodes": [], "edges": []}
        keep, layer = {target}, {target}
        for _ in range(depth):
            layer = {n for x in layer for n in (*self.graph.predecessors(x), *self.graph.successors(x))} - keep
            keep |= layer
        sub = self.graph.subgraph(keep)
        nodes = [{"id": n, "label": d.get("label", n), "is_target": n == target,
                  "provenance": d.get("provenance", "unknown"),
                  "allowlisted_merchant": bool(d.get("allowlisted_merchant"))} for n, d in sub.nodes(data=True)]
        edges = [{"from": u, "to": v, "amount": round(sum(t["amount_pyusd"] for t in d["transfers"]), 2),
                  "token": "PYUSD", "transfer_count": len(d["transfers"]),
                  "first_timestamp": min(t["timestamp"] for t in d["transfers"])} for u, v, d in sub.edges(data=True)]
        return {"nodes": nodes, "edges": edges}
