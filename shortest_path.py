"""
shortest_path.py - Shortest Path Algorithms: Dijkstra, Bellman-Ford, and Floyd-Warshall.

Manual implementations:
1. Dijkstra's Algorithm (Single-Source Shortest Path using min-heap / heapq):
   Time Complexity: O((|V| + |E|) log |V|)
   Space Complexity: O(|V| + |E|)
   Supports detailed step-by-step telemetry for algorithm animation.
2. Bellman-Ford Algorithm (Single-Source Shortest Path via Edge Relaxation):
   Time Complexity: O(|V| * |E|)
   Space Complexity: O(|V|)
   Supports pass-by-pass relaxation telemetry and negative cycle detection.
3. Floyd-Warshall Algorithm (All-Pairs Shortest Path via Dynamic Programming):
   Time Complexity: O(|V|^3)
   Space Complexity: O(|V|^2)
"""

from __future__ import annotations
import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from graph_utils import CloudNetwork


@dataclass
class DijkstraStep:
    """Telemetry information for one step in Dijkstra's algorithm."""
    step_num: int
    action: str                        # "SETTLE_NODE", "RELAX_EDGE", "SKIP_SETTLED", "NO_IMPROVEMENT"
    current_node: int                  # Current settled vertex u
    current_node_name: str
    current_dist: float
    neighbor_node: Optional[int] = None # Candidate vertex v
    neighbor_node_name: Optional[str] = None
    edge_latency: Optional[float] = None
    relaxed: bool = False
    reason: str = ""
    distances: Dict[int, float] = field(default_factory=dict)
    visited_nodes: List[int] = field(default_factory=list)
    active_tree_edges: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class DijkstraResult:
    """Telemetry and outcome of single-source Dijkstra run."""
    source: int
    distances: Dict[int, float]
    predecessors: Dict[int, Optional[int]]
    visit_order: List[int]
    steps: List[DijkstraStep] = field(default_factory=list)
    time_complexity: str = "O((|V| + |E|) log |V|)"
    space_complexity: str = "O(|V| + |E|)"

    def get_path_to(self, target: int) -> List[int]:
        """Reconstructs the shortest path from source to target using predecessor pointers."""
        if target not in self.distances or math.isinf(self.distances[target]):
            return []
        path = []
        curr: Optional[int] = target
        while curr is not None:
            path.append(curr)
            if curr == self.source:
                break
            curr = self.predecessors.get(curr)
        path.reverse()
        return path if (path and path[0] == self.source) else []


def dijkstra_shortest_path(network: CloudNetwork, source: int) -> DijkstraResult:
    """
    Computes shortest paths from a single source node to all other nodes in the network
    using Dijkstra's algorithm with a priority queue (min-heap).
    Records step-by-step exploration telemetry for live animation.
    """
    adj = network.adjacency_list()
    distances: Dict[int, float] = {node: float("inf") for node in network.servers}
    predecessors: Dict[int, Optional[int]] = {node: None for node in network.servers}
    visit_order: List[int] = []
    steps: List[DijkstraStep] = []

    if source not in network.servers:
        return DijkstraResult(source, distances, predecessors, visit_order, steps)

    distances[source] = 0.0
    # Min-heap items: (distance, node_id)
    pq: List[Tuple[float, int]] = [(0.0, source)]
    visited = set()

    step_counter = 0

    while pq:
        d, u = heapq.heappop(pq)

        if u in visited:
            continue
        visited.add(u)
        visit_order.append(u)

        u_name = network.servers[u].name
        tree_edges = [
            (min(predecessors[n], n), max(predecessors[n], n))
            for n in network.servers
            if predecessors[n] is not None
        ]

        step_counter += 1
        steps.append(
            DijkstraStep(
                step_num=step_counter,
                action="SETTLE_NODE",
                current_node=u,
                current_node_name=u_name,
                current_dist=d,
                reason=f"Settled vertex '{u_name}' with minimum tentative distance {d:.1f} ms.",
                distances=dict(distances),
                visited_nodes=list(visited),
                active_tree_edges=list(tree_edges),
            )
        )

        # Explore incident links
        for v, latency in adj.get(u, []):
            v_name = network.servers[v].name
            if v in visited:
                step_counter += 1
                steps.append(
                    DijkstraStep(
                        step_num=step_counter,
                        action="SKIP_SETTLED",
                        current_node=u,
                        current_node_name=u_name,
                        current_dist=d,
                        neighbor_node=v,
                        neighbor_node_name=v_name,
                        edge_latency=latency,
                        relaxed=False,
                        reason=f"Skipped link {u_name} ↔ {v_name}: vertex '{v_name}' already settled.",
                        distances=dict(distances),
                        visited_nodes=list(visited),
                        active_tree_edges=list(tree_edges),
                    )
                )
                continue

            new_dist = round(d + latency, 2)
            if new_dist < distances[v]:
                distances[v] = new_dist
                predecessors[v] = u
                heapq.heappush(pq, (new_dist, v))

                new_tree_edges = [
                    (min(predecessors[n], n), max(predecessors[n], n))
                    for n in network.servers
                    if predecessors[n] is not None
                ]

                step_counter += 1
                steps.append(
                    DijkstraStep(
                        step_num=step_counter,
                        action="RELAX_EDGE",
                        current_node=u,
                        current_node_name=u_name,
                        current_dist=d,
                        neighbor_node=v,
                        neighbor_node_name=v_name,
                        edge_latency=latency,
                        relaxed=True,
                        reason=f"Relaxed link {u_name} ↔ {v_name}: updated shortest distance to '{v_name}' = {new_dist:.1f} ms (via {u_name}).",
                        distances=dict(distances),
                        visited_nodes=list(visited),
                        active_tree_edges=list(new_tree_edges),
                    )
                )
            else:
                step_counter += 1
                steps.append(
                    DijkstraStep(
                        step_num=step_counter,
                        action="NO_IMPROVEMENT",
                        current_node=u,
                        current_node_name=u_name,
                        current_dist=d,
                        neighbor_node=v,
                        neighbor_node_name=v_name,
                        edge_latency=latency,
                        relaxed=False,
                        reason=f"No update for link {u_name} ↔ {v_name}: current distance to '{v_name}' ({distances[v]:.1f} ms) <= {new_dist:.1f} ms.",
                        distances=dict(distances),
                        visited_nodes=list(visited),
                        active_tree_edges=list(tree_edges),
                    )
                )

    return DijkstraResult(
        source=source,
        distances=distances,
        predecessors=predecessors,
        visit_order=visit_order,
        steps=steps,
    )


@dataclass
class BellmanFordStep:
    """Telemetry information for one edge relaxation step in Bellman-Ford."""
    step_num: int
    pass_num: int
    u: int
    v: int
    u_name: str
    v_name: str
    latency: float
    relaxed: bool
    reason: str
    updated_node: Optional[int] = None
    new_dist: Optional[float] = None
    distances: Dict[int, float] = field(default_factory=dict)
    active_tree_edges: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class BellmanFordResult:
    """Outcome of Bellman-Ford shortest path run."""
    source: int
    distances: Dict[int, float]
    predecessors: Dict[int, Optional[int]]
    has_negative_cycle: bool = False
    negative_cycle_nodes: List[int] = field(default_factory=list)
    passes_run: int = 1
    steps: List[BellmanFordStep] = field(default_factory=list)
    time_complexity: str = "O(|V| * |E|)"
    space_complexity: str = "O(|V|)"

    @property
    def relaxation_passes(self) -> int:
        return self.passes_run

    def get_path_to(self, target: int) -> List[int]:
        """Reconstructs the shortest path from source to target using predecessor pointers."""
        if target not in self.distances or math.isinf(self.distances[target]) or self.distances[target] == float("-inf"):
            return []
        path = []
        curr: Optional[int] = target
        visited = set()
        while curr is not None:
            if curr in visited:
                break
            visited.add(curr)
            path.append(curr)
            if curr == self.source:
                break
            curr = self.predecessors.get(curr)
        path.reverse()
        return path if (path and path[0] == self.source) else []


def bellman_ford_shortest_path(network: CloudNetwork, source: int) -> BellmanFordResult:
    """
    Computes single-source shortest paths using the manual Bellman-Ford algorithm.
    Relaxes all edges up to |V| - 1 times with early termination.
    Executes an extra |V|-th pass for negative cycle detection.
    When a negative cycle is detected, recovers the cycle nodes and marks reachable nodes as -infinity.
    """
    node_ids = list(network.servers.keys())
    distances: Dict[int, float] = {node: float("inf") for node in node_ids}
    predecessors: Dict[int, Optional[int]] = {node: None for node in node_ids}
    steps: List[BellmanFordStep] = []

    if source not in network.servers:
        return BellmanFordResult(source, distances, predecessors, steps=steps)

    distances[source] = 0.0
    directed_edges = network.directed_edges()
    max_passes = max(1, len(node_ids) - 1)
    passes_run = 0
    step_counter = 0

    # Relaxation passes: 1 to |V| - 1
    for pass_idx in range(1, max_passes + 1):
        passes_run = pass_idx
        updated_in_pass = False

        for u, v, lat in directed_edges:
            if u not in network.servers or v not in network.servers:
                continue
            u_name = network.servers[u].name
            v_name = network.servers[v].name

            # Check relaxation u -> v
            if distances[u] != float("inf") and distances[u] + lat < distances[v]:
                distances[v] = round(distances[u] + lat, 2)
                predecessors[v] = u
                updated_in_pass = True
                step_counter += 1

                tree_edges = [
                    (min(predecessors[n], n), max(predecessors[n], n))
                    for n in node_ids
                    if predecessors[n] is not None
                ]
                steps.append(
                    BellmanFordStep(
                        step_num=step_counter,
                        pass_num=pass_idx,
                        u=u,
                        v=v,
                        u_name=u_name,
                        v_name=v_name,
                        latency=lat,
                        relaxed=True,
                        updated_node=v,
                        new_dist=distances[v],
                        reason=f"Pass {pass_idx}: Relaxed {u_name} → {v_name} (wt={lat:.1f} ms). Tentative dist to '{v_name}' = {distances[v]:.1f} ms.",
                        distances=dict(distances),
                        active_tree_edges=list(tree_edges),
                    )
                )

        # Early termination if no relaxations happened in this pass
        if not updated_in_pass:
            break

    # |V|-th pass: Check for negative weight cycles
    has_neg_cycle = False
    cycle_nodes: List[int] = []
    cycle_start: Optional[int] = None

    for u, v, lat in directed_edges:
        if distances[u] != float("inf") and distances[u] + lat < distances[v]:
            has_neg_cycle = True
            cycle_start = v
            predecessors[v] = u
            break

    if has_neg_cycle and cycle_start is not None:
        # Walk predecessor pointers |V| times to guarantee entering the cycle
        curr = cycle_start
        for _ in range(len(node_ids)):
            if predecessors.get(curr) is not None:
                curr = predecessors[curr]

        # Trace the cycle path around until repeating
        cycle = [curr]
        p = predecessors.get(curr)
        while p is not None and p != curr and p not in cycle:
            cycle.append(p)
            p = predecessors.get(p)
        cycle.append(curr)
        cycle.reverse()
        cycle_nodes = cycle

        # Find all nodes reachable from the negative cycle and set distance to -inf
        adj = network.adjacency_list()
        affected_queue = list(set(cycle_nodes))
        affected_nodes = set(affected_queue)
        while affected_queue:
            curr_aff = affected_queue.pop(0)
            distances[curr_aff] = float("-inf")
            for nxt_node, _ in adj.get(curr_aff, []):
                if nxt_node not in affected_nodes:
                    affected_nodes.add(nxt_node)
                    affected_queue.append(nxt_node)

    return BellmanFordResult(
        source=source,
        distances=distances,
        predecessors=predecessors,
        has_negative_cycle=has_neg_cycle,
        negative_cycle_nodes=cycle_nodes,
        passes_run=passes_run,
        steps=steps,
    )


@dataclass
class FloydWarshallResult:
    """All-pairs shortest paths result and matrix utilities."""
    nodes: List[int]
    dist_matrix: np.ndarray  # shape (n, n)
    next_node_matrix: np.ndarray  # shape (n, n) with next hop node ID or -1
    node_to_idx: Dict[int, int]
    idx_to_node: Dict[int, int]
    has_negative_cycle: bool = False
    negative_cycle_nodes: List[int] = field(default_factory=list)
    intermediate_matrices: List[np.ndarray] = field(default_factory=list)
    time_complexity: str = "O(|V|^3)"
    space_complexity: str = "O(|V|^2)"

    def get_distance(self, u: int, v: int) -> float:
        """Returns shortest path latency between u and v."""
        if u not in self.node_to_idx or v not in self.node_to_idx:
            return float("inf")
        i = self.node_to_idx[u]
        j = self.node_to_idx[v]
        return float(self.dist_matrix[i, j])

    def get_path(self, u: int, v: int) -> List[int]:
        """Reconstructs shortest path from u to v using next_node matrix."""
        if u not in self.node_to_idx or v not in self.node_to_idx:
            return []
        if u == v:
            return [u]
        i = self.node_to_idx[u]
        j = self.node_to_idx[v]
        if math.isinf(self.dist_matrix[i, j]) or self.dist_matrix[i, j] == float("-inf"):
            return []

        path = [u]
        curr = i
        visited = set([curr])
        while curr != j:
            next_hop = self.next_node_matrix[curr, j]
            if next_hop == -1 or next_hop in visited:
                return []
            visited.add(next_hop)
            curr = next_hop
            path.append(self.idx_to_node[curr])
        return path

    def to_dataframe(self, network: CloudNetwork) -> pd.DataFrame:
        """Exports the latency matrix as a readable DataFrame with city names."""
        names = [network.servers[nid].name if nid in network.servers else f"ID {nid}" for nid in self.nodes]
        df = pd.DataFrame(self.dist_matrix, index=names, columns=names)
        return df.round(1)

    def dataframe_at_step(self, step_k: int, network: CloudNetwork) -> pd.DataFrame:
        """Exports the latency matrix at step k (0 <= step_k <= len(self.nodes))."""
        names = [network.servers[nid].name if nid in network.servers else f"ID {nid}" for nid in self.nodes]
        if self.intermediate_matrices and 0 <= step_k < len(self.intermediate_matrices):
            mat = self.intermediate_matrices[step_k]
        else:
            mat = self.dist_matrix
        df = pd.DataFrame(mat, index=names, columns=names)
        return df.round(1)


def floyd_warshall_all_pairs(network: CloudNetwork) -> FloydWarshallResult:
    """
    Computes all-pairs shortest path matrix using Floyd-Warshall Dynamic Programming.
    DP Recurrence: dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j]) for k in 0..n-1.
    Detects negative cycles when any diagonal element dist[i, i] < 0.
    """
    node_ids = sorted(list(network.servers.keys()))
    n = len(node_ids)

    node_to_idx: Dict[int, int] = {node_ids[i]: i for i in range(n)}
    idx_to_node: Dict[int, int] = {i: node_ids[i] for i in range(n)}

    # Distance and predecessor next-hop matrices
    dist = np.full((n, n), fill_value=float("inf"), dtype=float)
    next_node = np.full((n, n), fill_value=-1, dtype=int)

    # Base cases: dist(v, v) = 0
    for i in range(n):
        dist[i, i] = 0.0
        next_node[i, i] = i

    # Base cases: direct edge latencies (respecting directed links)
    for link in network.links:
        if link.u in node_to_idx and link.v in node_to_idx:
            u_idx = node_to_idx[link.u]
            v_idx = node_to_idx[link.v]
            if link.latency < dist[u_idx, v_idx]:
                dist[u_idx, v_idx] = link.latency
                next_node[u_idx, v_idx] = v_idx
            if not link.directed:
                if link.latency < dist[v_idx, u_idx]:
                    dist[v_idx, u_idx] = link.latency
                    next_node[v_idx, u_idx] = u_idx

    intermediate_matrices = [dist.copy()]

    # DP Triple Loop: k is intermediate vertex
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] != float("inf") and dist[k, j] != float("inf"):
                    alt = dist[i, k] + dist[k, j]
                    if alt < dist[i, j]:
                        dist[i, j] = round(alt, 2)
                        next_node[i, j] = next_node[i, k]
        intermediate_matrices.append(dist.copy())

    # Negative cycle check: any dist[i, i] < 0
    neg_nodes = [node_ids[i] for i in range(n) if dist[i, i] < 0.0]
    has_neg_cycle = len(neg_nodes) > 0

    return FloydWarshallResult(
        nodes=node_ids,
        dist_matrix=dist,
        next_node_matrix=next_node,
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node,
        has_negative_cycle=has_neg_cycle,
        negative_cycle_nodes=neg_nodes,
        intermediate_matrices=intermediate_matrices,
    )
