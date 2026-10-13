"""
tests/test_algorithms.py - Comprehensive Unit Tests for Cloud Task Placement Simulator.

Verifies:
1. Manual Kruskal MST against networkx.minimum_spanning_tree on random graphs.
2. Manual Dijkstra single-source shortest path against networkx.single_source_dijkstra.
3. Manual Floyd-Warshall all-pairs shortest path against networkx.all_pairs_dijkstra_path_length.
4. Knapsack DP optimal placement against Brute-Force enumerator.
5. Edge cases: disconnected graph handling, k > n, capacity deficit, capacity reduction demo.
"""

import math
import random
import pytest
import networkx as nx
from graph_utils import (
    CloudNetwork,
    Server,
    Link,
    Client,
    create_sample_network,
    generate_random_network,
    create_dijkstra_vs_bellman_network,
    create_negative_cycle_network,
    create_normal_network_5,
)
from mst import kruskal_mst, prim_mst, UnionFind
import plotly.graph_objects as go
from shortest_path import dijkstra_shortest_path, bellman_ford_shortest_path, floyd_warshall_all_pairs
from decision_tree import build_decision_tree, draw_decision_tree_plotly
from placement_dp import (
    Task,
    get_default_tasks,
    get_default_demo_tasks,
    get_greedy_trap_tasks,
    get_decision_tree_preset_tasks,
    solve_knapsack_01_dp,
    solve_knapsack_bruteforce,
    evaluate_all_servers_placement,
)
from baselines import (
    solve_knapsack_greedy_value,
    solve_knapsack_greedy_ratio,
    solve_knapsack_random,
    run_knapsack_comparison,
)


# =====================================================================
# 1. Union-Find & Kruskal vs NetworkX MST Tests
# =====================================================================

def test_union_find_operations():
    """Verifies Disjoint Set Union operations with path compression and rank."""
    uf = UnionFind([0, 1, 2, 3, 4])
    assert uf.num_components == 5
    assert not uf.connected(0, 1)

    assert uf.union(0, 1) is True
    assert uf.connected(0, 1)
    assert uf.num_components == 4

    assert uf.union(1, 2) is True
    assert uf.connected(0, 2)
    assert uf.num_components == 3

    # Attempting to union already connected elements returns False (cycle prevention)
    assert uf.union(0, 2) is False
    assert uf.num_components == 3


@pytest.mark.parametrize("seed", [10, 42, 99])
def test_kruskal_against_networkx(seed):
    """
    Compares manual Kruskal MST with NetworkX built-in MST across multiple random graphs.
    Verifies that total MST cost and tree edge count match exactly.
    """
    net = generate_random_network(n=8, edge_density=0.4, seed=seed)

    # 1. Run manual Kruskal
    res = kruskal_mst(net)
    assert res.is_spanning is True
    assert len(res.mst_edges) == net.num_servers - 1

    # 2. Build equivalent NetworkX graph
    G = nx.Graph()
    for s in net.servers.values():
        G.add_node(s.id)
    for l in net.links:
        G.add_edge(l.u, l.v, weight=l.latency)

    nx_mst = nx.minimum_spanning_tree(G, weight="weight")
    nx_cost = sum(d["weight"] for _, _, d in nx_mst.edges(data=True))

    assert math.isclose(res.total_cost, nx_cost, abs_tol=1e-2), (
        f"Kruskal total cost {res.total_cost} != NetworkX cost {nx_cost}"
    )


@pytest.mark.parametrize("seed", [10, 42, 99])
def test_prim_against_networkx_and_kruskal(seed):
    """
    Compares manual Prim MST with Kruskal MST and NetworkX built-in MST.
    Verifies total MST cost and tree edge count.
    """
    net = generate_random_network(n=8, edge_density=0.4, seed=seed)

    kruskal_res = kruskal_mst(net)
    prim_res = prim_mst(net)

    assert prim_res.is_spanning is True
    assert len(prim_res.mst_edges) == net.num_servers - 1

    # Both Kruskal and Prim must produce identical total minimum spanning cost
    assert math.isclose(prim_res.total_cost, kruskal_res.total_cost, abs_tol=1e-2), (
        f"Prim total cost {prim_res.total_cost} != Kruskal cost {kruskal_res.total_cost}"
    )

    # Compare with NetworkX
    G = nx.Graph()
    for s in net.servers.values():
        G.add_node(s.id)
    for l in net.links:
        G.add_edge(l.u, l.v, weight=l.latency)

    nx_mst = nx.minimum_spanning_tree(G, weight="weight")
    nx_cost = sum(d["weight"] for _, _, d in nx_mst.edges(data=True))
    assert math.isclose(prim_res.total_cost, nx_cost, abs_tol=1e-2)


# =====================================================================
# 2. Dijkstra vs NetworkX Shortest Path Tests
# =====================================================================

@pytest.mark.parametrize("source_node", [0, 3, 7])
def test_dijkstra_against_networkx(source_node):
    """
    Compares manual Dijkstra implementation against networkx.single_source_dijkstra.
    """
    net = create_sample_network()

    # 1. Manual Dijkstra
    dijkstra_res = dijkstra_shortest_path(net, source=source_node)

    # 2. NetworkX Dijkstra
    G = nx.Graph()
    for s in net.servers.values():
        G.add_node(s.id)
    for l in net.links:
        G.add_edge(l.u, l.v, weight=l.latency)

    nx_lengths, nx_paths = nx.single_source_dijkstra(G, source=source_node, weight="weight")

    for target in net.servers:
        manual_dist = dijkstra_res.distances[target]
        nx_dist = nx_lengths[target]
        assert math.isclose(manual_dist, nx_dist, abs_tol=1e-2), (
            f"Distance mismatch to {target}: manual={manual_dist}, nx={nx_dist}"
        )
        # Check path reconstruction validity
        manual_path = dijkstra_res.get_path_to(target)
        assert manual_path[0] == source_node
        assert manual_path[-1] == target
        # Compute cost along manual path
        path_cost = sum(
            G[manual_path[i]][manual_path[i + 1]]["weight"]
            for i in range(len(manual_path) - 1)
        )
        assert math.isclose(path_cost, nx_dist, abs_tol=1e-2)


@pytest.mark.parametrize("source_node", [0, 4, 8])
def test_bellman_ford_against_dijkstra(source_node):
    """
    Compares manual Bellman-Ford with manual Dijkstra.
    Verifies that distances and reconstructed paths match exactly.
    """
    net = create_sample_network()

    dijkstra_res = dijkstra_shortest_path(net, source=source_node)
    bf_res = bellman_ford_shortest_path(net, source=source_node)

    assert bf_res.has_negative_cycle is False

    for target in net.servers:
        d_dist = dijkstra_res.distances[target]
        bf_dist = bf_res.distances[target]
        assert math.isclose(d_dist, bf_dist, abs_tol=1e-2), (
            f"Bellman-Ford dist {bf_dist} != Dijkstra dist {d_dist} to target {target}"
        )

        bf_path = bf_res.get_path_to(target)
        if target == source_node:
            assert bf_path == [source_node]
        else:
            assert bf_path[0] == source_node
            assert bf_path[-1] == target


# =====================================================================
# 3. Floyd-Warshall vs NetworkX All-Pairs Shortest Path Tests
# =====================================================================

def test_floyd_warshall_against_networkx():
    """
    Compares manual Floyd-Warshall against NetworkX all-pairs Dijkstra.
    """
    net = create_sample_network()

    # 1. Manual Floyd-Warshall
    fw_res = floyd_warshall_all_pairs(net)

    # 2. NetworkX
    G = nx.Graph()
    for s in net.servers.values():
        G.add_node(s.id)
    for l in net.links:
        G.add_edge(l.u, l.v, weight=l.latency)

    nx_all = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))

    for u in net.servers:
        for v in net.servers:
            manual_dist = fw_res.get_distance(u, v)
            nx_dist = nx_all[u][v]
            assert math.isclose(manual_dist, nx_dist, abs_tol=1e-2), (
                f"Floyd-Warshall distance mismatch ({u}, {v}): manual={manual_dist}, nx={nx_dist}"
            )
            # Verify FW reconstructed path
            path = fw_res.get_path(u, v)
            if u == v:
                assert path == [u]
            else:
                assert path[0] == u
                assert path[-1] == v
                path_cost = sum(
                    G[path[i]][path[i + 1]]["weight"]
                    for i in range(len(path) - 1)
                )
                assert math.isclose(path_cost, nx_dist, abs_tol=1e-2)


# =====================================================================
# 4. 0/1 Knapsack DP vs Brute-Force Tests
# =====================================================================

@pytest.mark.parametrize("capacity", [30, 50, 75])
def test_knapsack_dp_against_bruteforce(capacity):
    """
    Verifies that 0/1 Knapsack DP finds the exact same optimal total value
    and valid total weight as full Brute-Force enumeration.
    """
    tasks = get_default_tasks()
    dp_res = solve_knapsack_01_dp(tasks, capacity)
    bf_res = solve_knapsack_bruteforce(tasks, capacity)

    assert dp_res.total_value == bf_res.total_value
    assert dp_res.total_weight <= capacity
    assert bf_res.total_weight <= capacity
    assert dp_res.total_weight == bf_res.total_weight


def test_greedy_trap_preset_dp_beats_greedy():
    """
    Verifies that for the classic trap preset (cap=50, Task A: 30w/60v, B: 25w/48v, C: 25w/48v),
    DP achieves 96 points while both Greedy heuristics are trapped at 60 points (+60% gain).
    """
    tasks, cap = get_greedy_trap_tasks()
    dp_res = solve_knapsack_01_dp(tasks, cap)
    g_val_res = solve_knapsack_greedy_value(tasks, cap)
    g_rat_res = solve_knapsack_greedy_ratio(tasks, cap)

    assert dp_res.total_value == 96
    assert g_val_res.total_value == 60
    assert g_rat_res.total_value == 60
    assert dp_res.total_value > g_val_res.total_value
    assert dp_res.total_value > g_rat_res.total_value


def test_server_eligibility_by_delay():
    """
    Verifies that servers are marked eligible only if Dijkstra delay <= max_delay.
    """
    net = create_sample_network()
    tasks = get_default_tasks()
    max_delay = 25.0
    eval_res = evaluate_all_servers_placement(net, entry_node_id=0, tasks=tasks, max_delay_ms=max_delay)

    for sid, ev in eval_res.server_evaluations.items():
        if ev.is_eligible:
            assert ev.delay_ms <= max_delay
            assert ev.knapsack is not None
        else:
            assert ev.delay_ms > max_delay or math.isinf(ev.delay_ms)
            assert ev.knapsack is None

    if eval_res.recommended_server_id is not None:
        rec_ev = eval_res.server_evaluations[eval_res.recommended_server_id]
        assert rec_ev.is_eligible
        assert rec_ev.delay_ms <= max_delay


# =====================================================================
# 5. Edge Cases & Validation Tests
# =====================================================================

def test_disconnected_graph_validation():
    """Verifies that disconnected graph triggers validation failure and components count."""
    net = create_sample_network()
    # Remove all international links connecting London/Frankfurt/Dubai to rest of Asia
    disconnected_links = [
        l for l in net.links
        if not (
            (l.u in [8, 9] or l.v in [8, 9])  # London, Frankfurt
        )
    ]
    net.links = disconnected_links
    is_valid, errors = net.validate(k=3)
    assert not is_valid
    assert any("disconnected" in err.lower() for err in errors)


def test_invalid_k_values():
    """Verifies that k <= 0 and k > n return explicit validation error."""
    net = create_sample_network()
    is_valid, errors = net.validate(k=15)
    assert not is_valid
    assert any("exceeds" in err.lower() for err in errors)

    is_valid, errors = net.validate(k=0)
    assert not is_valid
    assert any("greater than 0" in err.lower() for err in errors)


def test_capacity_deficit_validation():
    """Verifies error message when total capacity < total demand."""
    net = create_sample_network()
    # Artificially inflate client demands
    for c in net.clients:
        c.demand = 200
    is_valid, errors = net.validate(k=3)
    assert not is_valid
    assert any("less than total client demand" in err.lower() for err in errors)


def test_capacity_reduction_changes_placement():
    """
    Verifies that reducing server capacity decreases or adapts the achievable task value
    under the 0/1 Knapsack DP constraint.
    """
    tasks = get_default_tasks()
    res1 = solve_knapsack_01_dp(tasks, capacity=80)
    res2 = solve_knapsack_01_dp(tasks, capacity=30)

    assert res1.total_weight <= 80
    assert res2.total_weight <= 30
    assert res1.total_value >= res2.total_value
    assert len(res2.chosen_tasks) <= len(res1.chosen_tasks)


# =====================================================================
# 6. Decision Tree Engine Tests
# =====================================================================

def test_decision_tree_builder_and_preset():
    """
    Verifies the 3-task decision tree preset:
    A (w=1, v=1), B (w=2, v=7), C (w=3, v=11), Capacity = 5.
    - Expected best leaf: [B, C], Value: 18, Weight: 5.
    - Expected exceeded leaf: [A, B, C], Weight: 6 > 5, status='exceeded'.
    - Verifies match against 0/1 Knapsack DP.
    """
    from decision_tree import build_decision_tree
    from placement_dp import get_decision_tree_preset_tasks

    preset_tasks, cap = get_decision_tree_preset_tasks()
    tree_res = build_decision_tree(preset_tasks, capacity=cap, prune_exceeded=False)

    assert len(tree_res.nodes) == 15  # 2^(3+1) - 1 = 15 nodes
    assert tree_res.total_leaves == 8  # 2^3 = 8 leaves
    assert tree_res.best_leaf is not None
    assert tree_res.best_leaf.total_value == 18
    assert tree_res.best_leaf.total_weight == 5
    assert set(tree_res.best_leaf.selected_names) == {"B", "C"}
    assert tree_res.best_leaf.status == "optimal"

    # Find [A, B, C] leaf
    abc_leaf = next(n for n in tree_res.nodes if n.is_leaf and set(n.selected_names) == {"A", "B", "C"})
    assert abc_leaf.total_weight == 6
    assert abc_leaf.status == "exceeded"
    assert "exceeded" in abc_leaf.status_label.lower()

    # Compare against DP
    dp_res = solve_knapsack_01_dp(preset_tasks, capacity=cap)
    assert tree_res.best_leaf.total_value == dp_res.total_value
    assert set(tree_res.best_leaf.selected_names) == set(t.name for t in dp_res.chosen_tasks)


def test_decision_tree_pruning():
    """
    Verifies that enabling prune_exceeded stops expanding branches where weight > capacity,
    while preserving the identical global optimal value.
    """
    preset_tasks, cap = get_decision_tree_preset_tasks()
    unpruned = build_decision_tree(preset_tasks, capacity=2, prune_exceeded=False)
    pruned = build_decision_tree(preset_tasks, capacity=2, prune_exceeded=True)

    assert len(pruned.nodes) < len(unpruned.nodes)
    assert pruned.best_leaf is not None
    assert unpruned.best_leaf is not None
    assert pruned.best_leaf.total_value == unpruned.best_leaf.total_value


# =====================================================================
# 7. CHANGE C: Negative Weights, Bellman-Ford & Negative Cycles
# =====================================================================

def test_dijkstra_fails_and_bellman_ford_succeeds_preset1():
    """
    Verifies Preset 1:
    Directed edges: S->A = 4, S->B = 5, B->A = -3.
    Origin S (ID 0), Destination A (ID 1).
    - Dijkstra greedily settles A at distance 4 (path [S, A]).
      When B is popped at distance 5, link B->A (-3) is examined, but A is already settled.
      Therefore, Dijkstra outputs WRONG distance 4.
    - Bellman-Ford pass 1 relaxes S->B to 5, S->A to 4.
      Pass 2 relaxes B->A to 5 + (-3) = 2.
      Bellman-Ford outputs CORRECT distance 2 (path [S, B, A]).
    """
    net = create_dijkstra_vs_bellman_network()

    dijk_res = dijkstra_shortest_path(net, source=0)
    bf_res = bellman_ford_shortest_path(net, source=0)

    # Dijkstra gives wrong answer 4
    assert dijk_res.distances[1] == 4.0
    assert dijk_res.get_path_to(1) == [0, 1]

    # Bellman-Ford gives correct answer 2
    assert bf_res.distances[1] == 2.0
    assert bf_res.get_path_to(1) == [0, 2, 1]
    assert bf_res.has_negative_cycle is False

    # Explicit divergence assertion
    assert dijk_res.distances[1] > bf_res.distances[1]


def test_negative_cycle_detected_preset2():
    """
    Verifies Preset 2:
    Same graph as Preset 1 plus directed edge A->B = 1.
    Cycle B->A->B costs (-3) + 1 = -2 < 0 (negative cycle!).
    - Bellman-Ford detects the negative cycle in the V-th pass.
    - Accurately recovers cycle nodes (contains A and B).
    - Distances to reachable cycle nodes are set to -infinity.
    - No fake paths are returned for nodes affected by the negative cycle.
    - Floyd-Warshall also detects negative cycle.
    """
    net = create_negative_cycle_network()

    bf_res = bellman_ford_shortest_path(net, source=0)
    assert bf_res.has_negative_cycle is True
    assert set(bf_res.negative_cycle_nodes).issuperset({1, 2})

    # Nodes 1 and 2 are in / downstream of negative cycle -> -inf
    assert bf_res.distances[1] == float("-inf")
    assert bf_res.distances[2] == float("-inf")
    assert bf_res.get_path_to(1) == []
    assert bf_res.get_path_to(2) == []

    # Floyd-Warshall also detects the cycle
    fw_res = floyd_warshall_all_pairs(net)
    assert fw_res.has_negative_cycle is True


@pytest.mark.parametrize("seed", [11, 42, 77, 99])
def test_dijkstra_and_bellman_ford_agree_on_positive_graphs(seed):
    """
    Verifies that on positive graphs, Dijkstra and Bellman-Ford always compute
    identical shortest path distances and valid paths for all pairs.
    """
    net = generate_random_network(n=6, edge_density=0.5, seed=seed)
    source = min(net.servers.keys())

    dijk_res = dijkstra_shortest_path(net, source=source)
    bf_res = bellman_ford_shortest_path(net, source=source)

    assert bf_res.has_negative_cycle is False

    for target in net.servers:
        d_val = dijk_res.distances[target]
        b_val = bf_res.distances[target]
        if math.isinf(d_val):
            assert math.isinf(b_val)
        else:
            assert math.isclose(d_val, b_val, abs_tol=1e-2)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_knapsack_dp_vs_bruteforce_random_inputs(seed):
    """
    Generates random task weights and values and asserts that
    0/1 Knapsack DP matches the Brute-Force global optimum exactly.
    """
    rng = random.Random(seed)
    num_tasks = 6
    capacity = 25

    tasks = [
        Task(
            id=i + 1,
            name=f"Task-{i + 1}",
            weight=rng.randint(2, 14),
            value=rng.randint(5, 50),
        )
        for i in range(num_tasks)
    ]

    dp_res = solve_knapsack_01_dp(tasks, capacity)
    bf_res = solve_knapsack_bruteforce(tasks, capacity)

    assert dp_res.total_value == bf_res.total_value
    assert dp_res.total_weight <= capacity
    assert bf_res.total_weight <= capacity
    assert dp_res.total_weight == bf_res.total_weight


@pytest.mark.parametrize("seed", [10, 20, 30])
def test_tree_best_leaf_equals_dp(seed):
    """
    Verifies that the best valid leaf discovered by the Decision Tree
    exhaustively matches the 0/1 Knapsack DP optimal solution.
    """
    rng = random.Random(seed)
    tasks = [
        Task(
            id=i + 1,
            name=f"T{i + 1}",
            weight=rng.randint(1, 8),
            value=rng.randint(3, 25),
        )
        for i in range(4)
    ]
    capacity = 12

    tree_res = build_decision_tree(tasks, capacity=capacity, prune_exceeded=False)
    dp_res = solve_knapsack_01_dp(tasks, capacity=capacity)

    assert tree_res.best_leaf is not None
    assert tree_res.best_leaf.total_value == dp_res.total_value
    assert tree_res.best_leaf.total_weight == dp_res.total_weight


# =====================================================================
# 8. Edge Case Tests
# =====================================================================

def test_unreachable_node_infinity():
    """
    Verifies that an unreachable/isolated node has distance infinity
    and get_path_to returns an empty list.
    """
    net = CloudNetwork(
        servers={
            0: Server(0, "Gateway", 50, 100, 100),
            1: Server(1, "Connected", 50, 200, 100),
            2: Server(2, "Isolated", 50, 300, 100),
        },
        links=[Link(0, 1, 10.0, directed=False)],
        clients=[],
    )

    dijk_res = dijkstra_shortest_path(net, source=0)
    bf_res = bellman_ford_shortest_path(net, source=0)

    assert math.isinf(dijk_res.distances[2])
    assert math.isinf(bf_res.distances[2])
    assert dijk_res.get_path_to(2) == []
    assert bf_res.get_path_to(2) == []


def test_empty_task_list_knapsack():
    """Verifies that an empty task list safely returns 0 value and weight."""
    dp_res = solve_knapsack_01_dp([], capacity=50)
    assert dp_res.total_value == 0
    assert dp_res.total_weight == 0
    assert dp_res.chosen_tasks == []


def test_task_heavier_than_capacity():
    """Verifies that tasks heavier than capacity are skipped and 0 value is returned."""
    tasks = [Task(1, "Heavy-Task", weight=60, value=100)]
    dp_res = solve_knapsack_01_dp(tasks, capacity=50)
    assert dp_res.total_value == 0
    assert dp_res.total_weight == 0
    assert dp_res.chosen_tasks == []


def test_negative_undirected_edge_rejected():
    """
    Verifies that an undirected edge with negative weight is strictly rejected with:
    'A negative undirected edge is an automatic negative cycle. Make it one-way.'
    """
    net = CloudNetwork(
        servers={
            0: Server(0, "S0", 50, 100, 100),
            1: Server(1, "S1", 50, 200, 100),
        },
        links=[Link(0, 1, -5.0, directed=False)],
        clients=[],
    )
    is_valid, errors = net.validate(allow_negative_weights=True)
    assert not is_valid
    assert any("A negative undirected edge is an automatic negative cycle. Make it one-way." in err for err in errors)


def test_negative_weights_disabled_rejects_negative_edge():
    """
    Verifies that when 'Allow negative weights' is False, any negative weight edge is rejected.
    """
    net = CloudNetwork(
        servers={
            0: Server(0, "S0", 50, 100, 100),
            1: Server(1, "S1", 50, 200, 100),
        },
        links=[Link(0, 1, -5.0, directed=True)],
        clients=[],
    )
    is_valid, errors = net.validate(allow_negative_weights=False)
    assert not is_valid
    assert any("rejected" in err.lower() and "allow negative weights" in err.lower() for err in errors)


def test_placement_blocked_on_negative_cycle():
    """
    Verifies that when a reachable negative cycle exists, placement optimization is blocked
    and returns a graceful status message without crashing.
    """
    net = create_negative_cycle_network()
    tasks = get_default_tasks()
    eval_res = evaluate_all_servers_placement(
        network=net,
        entry_node_id=0,
        tasks=tasks,
        max_delay_ms=100.0,
        allow_negative_weights=True,
    )

    assert eval_res.recommended_server_id is None
    assert eval_res.eligible_count == 0
    assert "Placement blocked" in eval_res.status_message


def test_knapsack_tree_tab_preset_and_figure_rendering():
    """
    Verifies the exact requirements for the Knapsack Tree tab:
    - Default items: A (val 1, wt 1), B (val 7, wt 2), C (val 11, wt 3), capacity 5.
    - Expected result: best = [B, C], value 18, weight 5.
    - [A, B, C] is a red leaf (value 19, weight 6) with 'Maximum capacity exceeded'.
    - draw_decision_tree_plotly produces rounded box annotations with exact colors and text.
    """
    from decision_tree import build_decision_tree, draw_decision_tree_plotly
    import plotly.graph_objects as go

    items = [
        Task(id=1, name="A", weight=1, value=1),
        Task(id=2, name="B", weight=2, value=7),
        Task(id=3, name="C", weight=3, value=11),
    ]
    capacity = 5

    tree_res = build_decision_tree(items, capacity=capacity, prune_exceeded=False)

    # 1. Verification of tree structure and optimum
    assert tree_res.total_leaves == 8
    assert len(tree_res.nodes) == 15
    assert tree_res.best_leaf is not None
    assert set(tree_res.best_leaf.selected_names) == {"B", "C"}
    assert tree_res.best_leaf.total_value == 18
    assert tree_res.best_leaf.total_weight == 5

    # 2. Verification of [A, B, C] exceeded leaf
    abc_leaf = next(n for n in tree_res.nodes if n.is_leaf and set(n.selected_names) == {"A", "B", "C"})
    assert abc_leaf.total_weight == 6
    assert abc_leaf.total_value == 19
    assert abc_leaf.status == "exceeded"

    # 3. Verification of Figure rendering (circular nodes, short codes, hover details)
    fig = draw_decision_tree_plotly(tree_res)
    assert isinstance(fig, go.Figure)

    # Extract all annotations text
    ann_texts = [a.text for a in fig.layout.annotations if a.text is not None]
    # Extract hover + visible node text from traces
    trace_texts = []
    for tr in fig.data:
        if getattr(tr, "hovertext", None) is not None:
            trace_texts.extend(list(tr.hovertext))
        if getattr(tr, "text", None) is not None:
            t = tr.text
            trace_texts.extend(list(t) if isinstance(t, (list, tuple)) else [t])
    all_texts = ann_texts + [str(t) for t in trace_texts if t is not None]

    # Verify Root (short code ROOT / empty set, values in hover)
    assert any("ROOT" in t for t in all_texts)
    assert any("selected: []" in t and "Value:" in t for t in all_texts)

    # Verify Edge labels: short +A/-A badges, full "select A" kept for accessibility
    assert any("+A" in t for t in all_texts)
    assert any("-A" in t or "skip A" in t for t in all_texts)
    assert any("+B" in t or "select B" in t for t in all_texts)
    assert any("+C" in t or "select C" in t for t in all_texts)

    # Verify Leaf captions (in hover)
    assert any("Maximum value at capacity 5" in t for t in all_texts)
    assert any("Maximum capacity exceeded" in t for t in all_texts)

    # Verify Top-right setup table
    assert any("Capacity: C = 5" in t for t in all_texts)

    # Verify circular markers
    assert fig.data[0].marker.symbol == "circle" or "circle" in str(fig.data[0].marker.symbol)


def test_knapsack_tree_n_counts_and_hard_cap():
    """
    Verifies decision tree behavior across different item counts:
    - n <= 5 (e.g. n=4): builds and draws the full tree cleanly without warnings or errors.
    - n = 6: calling build_decision_tree directly raises ValueError due to the hard cap (>5 items).
    - n = 6 trimmed to first 5 items: builds and draws cleanly with 2^5 = 32 leaves.
    """
    # 1. n = 4 items: builds and draws full tree (2^4 = 16 leaves, 2^5 - 1 = 31 nodes)
    tasks_4 = [
        Task(id=1, name="T1", weight=2, value=3),
        Task(id=2, name="T2", weight=3, value=4),
        Task(id=3, name="T3", weight=4, value=5),
        Task(id=4, name="T4", weight=5, value=8),
    ]
    tree_res_4 = build_decision_tree(tasks_4, capacity=10, prune_exceeded=False)
    assert tree_res_4.total_leaves == 16
    assert len(tree_res_4.nodes) == 31
    assert tree_res_4.best_leaf is not None
    assert tree_res_4.best_leaf.total_weight <= 10

    fig_4 = draw_decision_tree_plotly(tree_res_4)
    assert isinstance(fig_4, go.Figure)
    assert len(fig_4.layout.annotations) > 0

    # 2. n = 6 items: calling builder directly raises ValueError (hard cap)
    tasks_6 = [
        Task(id=i, name=f"Item-{i}", weight=i, value=i * 2)
        for i in range(1, 7)
    ]
    with pytest.raises(ValueError, match="exceeding the hard cap of 5 tasks"):
        build_decision_tree(tasks_6, capacity=15)

    # 3. n = 6 trimmed to first 5 items: builds and draws successfully
    tasks_trimmed_5 = tasks_6[:5]
    tree_res_trimmed = build_decision_tree(tasks_trimmed_5, capacity=15, prune_exceeded=False)
    assert tree_res_trimmed.total_leaves == 32
    assert len(tree_res_trimmed.nodes) == 63
    assert tree_res_trimmed.best_leaf is not None

    fig_trimmed = draw_decision_tree_plotly(tree_res_trimmed)
    assert isinstance(fig_trimmed, go.Figure)


def test_normal_5_node_network_and_demo_tasks():
    """
    Verifies that the normal 5-node network is connected, valid, has Kruskal MST cost of 55 ms,
    and the 4 default tasks have sum of weights (75) > 50 producing at least one red overflow leaf.
    """
    net = create_normal_network_5()
    assert net.num_servers == 5
    assert net.is_connected()
    is_valid, errs = net.validate(allow_negative_weights=False)
    assert is_valid, errs

    # MST check
    mst_res = kruskal_mst(net)
    assert len(mst_res.mst_edges) == 4
    assert mst_res.total_cost == 55.0

    # Default tasks check
    tasks = get_default_demo_tasks()
    assert len(tasks) == 4
    total_wt = sum(t.weight for t in tasks)
    assert total_wt == 75

    # On server capacity 50:
    tree_res = build_decision_tree(tasks, capacity=50, prune_exceeded=False)
    assert tree_res.total_leaves == 16
    assert tree_res.exceeded_leaves_count >= 1
    # Check that all 4 tasks together exceed capacity 50
    all_leaf = next(n for n in tree_res.nodes if n.is_leaf and len(n.selected_tasks) == 4)
    assert all_leaf.status == "exceeded"
    assert all_leaf.total_weight == 75

    # Best valid leaf matches DP
    dp_res = solve_knapsack_01_dp(tasks, capacity=50)
    assert tree_res.best_leaf is not None
    assert tree_res.best_leaf.total_value == dp_res.total_value
    assert tree_res.best_leaf.total_weight == dp_res.total_weight





