"""
mst.py - Minimum Spanning Tree Backbone using Kruskal's Algorithm.

Manual implementation of Kruskal's algorithm with Disjoint-Set Union (Union-Find)
supporting path compression and union-by-rank.
Provides detailed step-by-step telemetry for algorithm animation and educational inspection.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from graph_utils import CloudNetwork, Link, Server


class UnionFind:
    """
    Disjoint Set Union (DSU) / Union-Find data structure with:
    1. Path compression in find()
    2. Union by rank in union()

    Amortized Time Complexity: O(alpha(V)) per operation, where alpha is the inverse Ackermann function.
    Space Complexity: O(V)
    """

    def __init__(self, elements: List[int]):
        self.parent: Dict[int, int] = {e: e for e in elements}
        self.rank: Dict[int, int] = {e: 0 for e in elements}
        self.num_components: int = len(elements)

    def find(self, i: int) -> int:
        """Find the root representative of element i with path compression."""
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])  # Path compression
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        """
        Merge sets containing i and j using union by rank.
        Returns True if sets were disjoint and successfully merged, False if already connected.
        """
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i == root_j:
            return False  # Already in the same set, would introduce a cycle

        # Union by rank: attach smaller rank tree under higher rank tree
        if self.rank[root_i] < self.rank[root_j]:
            self.parent[root_i] = root_j
        elif self.rank[root_i] > self.rank[root_j]:
            self.parent[root_j] = root_i
        else:
            self.parent[root_j] = root_i
            self.rank[root_i] += 1

        self.num_components -= 1
        return True

    def connected(self, i: int, j: int) -> bool:
        return self.find(i) == self.find(j)


@dataclass
class MSTStep:
    """Telemetry information for one edge evaluation in MST algorithms."""
    step_num: int
    u: int
    v: int
    u_name: str
    v_name: str
    latency: float
    accepted: bool
    reason: str
    components_count: int
    running_cost: float
    mst_edges_so_far: List[Tuple[int, int, float]] = field(default_factory=list)


# Alias for backward compatibility
KruskalStep = MSTStep


@dataclass
class MSTResult:
    """Outcome of MST algorithm (Kruskal or Prim)."""
    algorithm: str
    mst_edges: List[Tuple[int, int, float]]
    total_cost: float
    is_spanning: bool
    steps: List[MSTStep]
    time_complexity: str = "O(|E| log |E|)"
    space_complexity: str = "O(|V| + |E|)"


def kruskal_mst(network: CloudNetwork) -> MSTResult:
    """Computes MST using Kruskal's algorithm with Union-Find."""
    node_ids = list(network.servers.keys())
    if len(node_ids) <= 1:
        return MSTResult(
            algorithm="Kruskal",
            mst_edges=[],
            total_cost=0.0,
            is_spanning=True,
            steps=[],
            time_complexity="O(|E| log |E|)",
        )

    edges = network.unique_edges()
    edges.sort(key=lambda edge: edge[2])

    uf = UnionFind(node_ids)
    mst_edges: List[Tuple[int, int, float]] = []
    total_cost: float = 0.0
    steps: List[MSTStep] = []
    target_count = len(node_ids) - 1

    for idx, (u, v, lat) in enumerate(edges, start=1):
        u_name = network.servers[u].name
        v_name = network.servers[v].name

        root_u = uf.find(u)
        root_v = uf.find(v)

        if root_u != root_v:
            uf.union(u, v)
            mst_edges.append((u, v, lat))
            total_cost += lat
            accepted = True
            reason = f"Accepted: '{u_name}' and '{v_name}' in disjoint sets; merged."
        else:
            accepted = False
            reason = f"Rejected: Link '{u_name}' ↔ '{v_name}' forms a cycle."

        steps.append(
            MSTStep(
                step_num=idx,
                u=u,
                v=v,
                u_name=u_name,
                v_name=v_name,
                latency=lat,
                accepted=accepted,
                reason=reason,
                components_count=uf.num_components,
                running_cost=round(total_cost, 2),
                mst_edges_so_far=list(mst_edges),
            )
        )

        if len(mst_edges) == target_count:
            break

    is_spanning = len(mst_edges) == target_count and uf.num_components == 1

    return MSTResult(
        algorithm="Kruskal",
        mst_edges=mst_edges,
        total_cost=round(total_cost, 2),
        is_spanning=is_spanning,
        steps=steps,
        time_complexity="O(|E| log |E|)",
    )


def prim_mst(network: CloudNetwork, start_node: Optional[int] = None) -> MSTResult:
    """
    Computes MST using Prim's algorithm with a priority queue (min-heap).
    
    Time Complexity: O(|E| log |V|)
    Space Complexity: O(|V| + |E|)
    """
    node_ids = sorted(list(network.servers.keys()))
    if len(node_ids) <= 1:
        return MSTResult(
            algorithm="Prim",
            mst_edges=[],
            total_cost=0.0,
            is_spanning=True,
            steps=[],
            time_complexity="O(|E| log |V|)",
        )

    import heapq
    adj = network.adjacency_list()
    root = start_node if (start_node is not None and start_node in network.servers) else node_ids[0]

    visited = {root}
    # Min-heap items: (latency, u_in_tree, v_candidate)
    pq: List[Tuple[float, int, int]] = []
    for neighbor, lat in adj.get(root, []):
        heapq.heappush(pq, (lat, root, neighbor))

    mst_edges: List[Tuple[int, int, float]] = []
    total_cost: float = 0.0
    steps: List[MSTStep] = []
    step_num = 0
    target_count = len(node_ids) - 1

    while pq and len(mst_edges) < target_count:
        lat, u, v = heapq.heappop(pq)
        step_num += 1

        u_name = network.servers[u].name
        v_name = network.servers[v].name

        if v in visited:
            steps.append(
                MSTStep(
                    step_num=step_num,
                    u=u,
                    v=v,
                    u_name=u_name,
                    v_name=v_name,
                    latency=lat,
                    accepted=False,
                    reason=f"Rejected: Vertex '{v_name}' already part of tree (cycle avoided).",
                    components_count=len(node_ids) - len(visited),
                    running_cost=round(total_cost, 2),
                    mst_edges_so_far=list(mst_edges),
                )
            )
            continue

        # Accept edge connecting tree to new vertex
        visited.add(v)
        mst_edges.append((min(u, v), max(u, v), lat))
        total_cost += lat

        steps.append(
            MSTStep(
                step_num=step_num,
                u=u,
                v=v,
                u_name=u_name,
                v_name=v_name,
                latency=lat,
                accepted=True,
                reason=f"Accepted: Minimum frontier edge to connect '{v_name}' to tree.",
                components_count=len(node_ids) - len(visited),
                running_cost=round(total_cost, 2),
                mst_edges_so_far=list(mst_edges),
            )
        )

        # Push new frontier edges
        for neighbor, weight in adj.get(v, []):
            if neighbor not in visited:
                heapq.heappush(pq, (weight, v, neighbor))

    is_spanning = len(visited) == len(node_ids)

    return MSTResult(
        algorithm="Prim",
        mst_edges=mst_edges,
        total_cost=round(total_cost, 2),
        is_spanning=is_spanning,
        steps=steps,
        time_complexity="O(|E| log |V|)",
    )

