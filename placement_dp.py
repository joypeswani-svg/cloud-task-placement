"""
placement_dp.py - Task Placement Optimization using Classic 0/1 Knapsack Dynamic Programming.

Algorithmic Paradigm:
---------------------
Dynamic Programming (Classic 0/1 Knapsack Problem Maximizing Total Value).

Problem Formulation:
--------------------
- Items: A collection of n tasks T = {t_1, t_2, ..., t_n}.
  Each task i has:
    1. name: Human-readable task descriptor (e.g., "Auth Service", "ML Inference").
    2. weight w_i: Resource units needed (CPU cores / RAM GB), w_i > 0.
    3. value v_i: Priority score / business profit earned if scheduled, v_i > 0.

- Bag: A single selected physical server s with compute capacity C = cap(s).
- Constraint: Total weight of scheduled tasks must not exceed server capacity C.
  Each task is binary (0/1): it is either taken in whole or skipped entirely.
  Tasks cannot be fractionally divided or duplicated.

- Objective: Maximize the total value of scheduled tasks:
    maximize sum_{i in S} v_i  subject to  sum_{i in S} w_i <= C  where S subseteq {1, ..., n}.

Connecting to the Cloud Network Graph:
--------------------------------------
1. A user entry node (gateway server) is chosen where workloads enter the cloud.
2. Shortest-path routing delays to all candidate servers are computed manually
   using Dijkstra's algorithm (or Bellman-Ford algorithm).
3. A server is ELIGIBLE if and only if:
       delay(entry_node, server) <= max_allowed_delay_ms
4. The 0/1 Knapsack DP is evaluated on each eligible server using that server's capacity C.
5. The system recommends the server achieving the MAXIMUM total task value.
   Ties are broken by lowest network delay from the user entry node.

Dynamic Programming Recurrence & Mathematical Derivation:
----------------------------------------------------------
State Definition:
    dp[i][c] represents the maximum total value obtainable using a subset
    of the first i tasks {t_1, ..., t_i} with a total weight not exceeding c,
    for 0 <= i <= n and 0 <= c <= C.

Base Cases:
    dp[0][c] = 0   for all 0 <= c <= C   (0 tasks available yields 0 value)
    dp[i][0] = 0   for all 0 <= i <= n   (0 capacity can hold 0 weight)

Recurrence Relation:
    For task i with weight w_i and value v_i:
    For each capacity c from 0 to C:

        Case 1: Task i is heavier than remaining capacity (w_i > c):
            Task i cannot fit in the bag.
            dp[i][c] = dp[i-1][c]   (Must SKIP task i)

        Case 2: Task i fits in the bag (w_i <= c):
            We compare two mutually exclusive choices:
            Choice A (SKIP): Do not schedule task i.
                Value = dp[i-1][c]
            Choice B (TAKE): Schedule task i, occupying w_i capacity and adding v_i value.
                Value = dp[i-1][c - w_i] + v_i

            dp[i][c] = max(dp[i-1][c], dp[i-1][c - w_i] + v_i)

Backtracking Procedure:
-----------------------
Start at cell (i = n, c = C):
    While i > 0 and c > 0:
        If dp[i][c] == dp[i-1][c]:
            # Task i was skipped
            i -= 1
        Else:
            # Task i was taken!
            Record task i as chosen
            c -= w_i
            i -= 1

Complexity Analysis:
--------------------
Time Complexity:
    - Number of states = (n + 1) rows * (C + 1) columns.
    - Each cell transition requires O(1) elementary comparisons and additions.
    - Total Time Complexity: O(n * C).
    - For n = 8 tasks and server capacity C = 60:
      Total cell evaluations = 9 * 61 = 549 operations -> runs in < 0.5 milliseconds!

Space Complexity:
    - DP table dimensions: (n + 1) x (C + 1) integer array.
    - Total Space Complexity: O(n * C) memory.
    - Full table is retained to reconstruct the exact optimal subset of tasks
      and render the interactive step-by-step visual audit trail.
"""

from __future__ import annotations
import copy
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
import pandas as pd
from graph_utils import CloudNetwork, Server
from shortest_path import dijkstra_shortest_path, DijkstraResult, bellman_ford_shortest_path, BellmanFordResult


@dataclass
class Task:
    """A computational task to be placed on a server."""
    id: int
    name: str
    weight: int   # CPU / RAM capacity units required (w_i > 0)
    value: int    # Priority / Profit points earned (v_i > 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "weight": self.weight,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Task:
        return cls(
            id=int(d["id"]),
            name=str(d["name"]),
            weight=int(d["weight"]),
            value=int(d["value"]),
        )


def get_default_tasks() -> List[Task]:
    """Returns 8 realistic, pre-filled cloud tasks with varied weight/value profiles."""
    return [
        Task(id=1, name="Auth Service", weight=10, value=60),
        Task(id=2, name="Payment API", weight=15, value=100),
        Task(id=3, name="Search Indexer", weight=20, value=120),
        Task(id=4, name="Analytics Worker", weight=25, value=90),
        Task(id=5, name="Video Transcoder", weight=30, value=150),
        Task(id=6, name="ML Inference", weight=35, value=180),
        Task(id=7, name="Log Ingestion", weight=12, value=70),
        Task(id=8, name="Cache Warmer", weight=8, value=40),
    ]


def get_default_demo_tasks() -> List[Task]:
    """
    Returns the 4 default tasks for college demonstration:
    - Auth Service: weight 15, value 25
    - Payment API: weight 20, value 35
    - Search Indexer: weight 25, value 45
    - Worker Node: weight 15, value 30
    Total weight = 75. On a server with capacity 50, sum of weights exceeds capacity,
    guaranteeing at least one red (overflow) leaf in the decision tree.
    """
    return [
        Task(id=1, name="Auth Service", weight=15, value=25),
        Task(id=2, name="Payment API", weight=20, value=35),
        Task(id=3, name="Search Indexer", weight=25, value=45),
        Task(id=4, name="Worker Node", weight=15, value=30),
    ]



def get_greedy_trap_tasks() -> Tuple[List[Task], int]:
    """
    Returns a classic counterexample preset where Greedy heuristics fail
    and Dynamic Programming provably wins.

    Scenario: Server Capacity = 50
    - Task A: weight 30, value 60 (ratio = 2.0)
    - Task B: weight 25, value 48 (ratio = 1.92)
    - Task C: weight 25, value 48 (ratio = 1.92)

    Greedy (Highest Value First):
      Picks Task A (val=60, w=30). Remaining cap = 20. Neither B nor C fits.
      Total Value = 60, Total Weight = 30.

    Greedy (Value/Weight Ratio):
      Task A has highest ratio (2.0 > 1.92). Picks Task A. Same trap!
      Total Value = 60, Total Weight = 30.

    Dynamic Programming (Optimal):
      Picks Task B + Task C (weight = 25 + 25 = 50 <= 50).
      Total Value = 48 + 48 = 96!

    Outcome: DP achieves 96 vs Greedy's 60 (+60.0% improvement!).
    """
    tasks = [
        Task(id=1, name="Heavy Task A", weight=30, value=60),
        Task(id=2, name="Medium Task B", weight=25, value=48),
        Task(id=3, name="Medium Task C", weight=25, value=48),
    ]
    return tasks, 50


def get_decision_tree_preset_tasks() -> Tuple[List[Task], int]:
    """
    Returns the 3-task Decision Tree demonstration preset:
    A: (weight 1, value 1)
    B: (weight 2, value 7)
    C: (weight 3, value 11)
    Capacity: 5

    Expected:
    - Optimal valid leaf: [B, C], value 18, weight 5 (GREEN highlight).
    - Exceeded leaf: [A, B, C], value 19, weight 6 > 5 (RED highlight: 'Maximum capacity exceeded').
    """
    tasks = [
        Task(id=1, name="A", weight=1, value=1),
        Task(id=2, name="B", weight=2, value=7),
        Task(id=3, name="C", weight=3, value=11),
    ]
    return tasks, 5



@dataclass
class KnapsackStep:
    """Step-by-step telemetry record for filling a single cell dp[i][c]."""
    step_num: int
    task_idx: int         # 1-indexed task index i
    task_name: str
    capacity: int         # Column capacity c
    action: str           # "TAKE", "SKIP", or "CANNOT_FIT"
    skip_val: int         # dp[i-1][c]
    take_val: Optional[int] # dp[i-1][c - w_i] + v_i if w_i <= c else None
    cell_val: int         # Resulting dp[i][c]
    explanation: str      # Human-readable one-line rationale
    task_weight: int = 0
    task_value: int = 0

    @property
    def capacity_col(self) -> int:
        return self.capacity

    @property
    def current_value(self) -> int:
        return self.cell_val


@dataclass
class KnapsackResult:
    """Complete results from the 0/1 Knapsack Dynamic Programming solver."""
    tasks: List[Task]
    capacity: int
    dp_table: List[List[int]]                 # (n + 1) x (capacity + 1)
    chosen_tasks: List[Task]
    total_weight: int
    total_value: int
    capacity_used_pct: float
    backtrack_cells: List[Tuple[int, int]]   # Sequence of (i, c) cells visited in backtracking
    backtrack_chosen_indices: List[int]       # 1-indexed task indices chosen
    steps: List[KnapsackStep] = field(default_factory=list)
    time_elapsed_ms: float = 0.0

    def tasks_dataframe(self) -> pd.DataFrame:
        """Returns chosen tasks as an annotated DataFrame."""
        if not self.chosen_tasks:
            return pd.DataFrame(columns=["Task ID", "Task Name", "Weight (Units)", "Value (Points)", "Value/Weight Ratio"])
        records = []
        for t in self.chosen_tasks:
            records.append({
                "Task ID": t.id,
                "Task Name": t.name,
                "Weight (Units)": t.weight,
                "Value (Points)": t.value,
                "Value/Weight Ratio": round(t.value / t.weight, 2) if t.weight > 0 else 0.0,
            })
        return pd.DataFrame(records)

    def dp_dataframe(self, max_columns: int = 40) -> pd.DataFrame:
        """
        Converts the 2D DP table into an annotated pandas DataFrame for UI table view.
        Rows: 0 (Base Case) and 1..n (Tasks).
        Columns: Capacity states c in [0, capacity].
        """
        n = len(self.tasks)
        cap = max(0, self.capacity)
        if not self.dp_table:
            return pd.DataFrame()

        # Generate column indices
        if cap <= max_columns:
            col_indices = list(range(cap + 1))
            col_headers = [f"c={c}" for c in col_indices]
        else:
            stride = max(1, math.ceil(cap / (max_columns - 1)))
            col_indices = list(range(0, cap + 1, stride))
            if cap not in col_indices:
                col_indices.append(cap)
            col_headers = [f"c={c}" for c in col_indices]

        row_headers = ["Row 0: Base (No Tasks)"] + [
            f"Row {i}: {t.name} (w={t.weight}, v={t.value})"
            for i, t in enumerate(self.tasks, start=1)
        ]

        table_data = []
        for i in range(n + 1):
            row = [self.dp_table[i][c] for c in col_indices]
            table_data.append(row)

        return pd.DataFrame(table_data, index=row_headers, columns=col_headers)


def solve_knapsack_01_dp(tasks: List[Task], capacity: int) -> KnapsackResult:
    """
    Solves the classic 0/1 Knapsack Problem using Dynamic Programming to MAXIMIZE total value.

    Recurrence:
        dp[i][c] = max(dp[i-1][c], dp[i-1][c - w_i] + v_i) if w_i <= c else dp[i-1][c]

    Base cases:
        dp[0][c] = 0 for all c in [0, C]
        dp[i][0] = 0 for all i in [0, n]

    Time Complexity:  O(n * C)
    Space Complexity: O(n * C)
    """
    start_time = time.perf_counter()
    n = len(tasks)
    C = max(0, int(capacity))

    # Initialize DP table of size (n + 1) x (C + 1) with 0s
    dp = [[0] * (C + 1) for _ in range(n + 1)]
    steps: List[KnapsackStep] = []
    step_num = 1

    # Fill table cell by cell
    for i in range(1, n + 1):
        t = tasks[i - 1]
        w_i = t.weight
        v_i = t.value

        for c in range(C + 1):
            skip_val = dp[i - 1][c]

            if w_i > c:
                # Task i is too heavy for capacity c; must skip
                dp[i][c] = skip_val
                action = "CANNOT_FIT"
                take_val = None
                exp = f"Task '{t.name}' (w={w_i}) exceeds capacity {c} → SKIP (value = {skip_val})"
            else:
                # Task i fits; compare SKIP vs TAKE
                take_val = dp[i - 1][c - w_i] + v_i
                if take_val > skip_val:
                    dp[i][c] = take_val
                    action = "TAKE"
                    exp = f"Fits '{t.name}' (w={w_i}, v={v_i}): TAKE ({take_val}) > SKIP ({skip_val}) → value = {take_val}"
                else:
                    dp[i][c] = skip_val
                    action = "SKIP"
                    exp = f"Fits '{t.name}' (w={w_i}, v={v_i}): SKIP ({skip_val}) >= TAKE ({take_val}) → value = {skip_val}"

            steps.append(KnapsackStep(
                step_num=step_num,
                task_idx=i,
                task_name=t.name,
                capacity=c,
                action=action,
                skip_val=skip_val,
                take_val=take_val,
                cell_val=dp[i][c],
                explanation=exp,
                task_weight=w_i,
                task_value=v_i,
            ))
            step_num += 1

    # Backtracking from cell (n, C) to recover chosen tasks
    curr_c = C
    chosen_tasks: List[Task] = []
    backtrack_cells: List[Tuple[int, int]] = [(n, curr_c)]
    backtrack_chosen_indices: List[int] = []

    for i in range(n, 0, -1):
        t = tasks[i - 1]
        w_i = t.weight
        v_i = t.value

        # If taking task i yielded the optimal value in dp[i][curr_c]
        if curr_c >= w_i and dp[i][curr_c] == dp[i - 1][curr_c - w_i] + v_i and dp[i][curr_c] > dp[i - 1][curr_c]:
            chosen_tasks.append(t)
            backtrack_chosen_indices.append(i)
            curr_c -= w_i
        elif curr_c >= w_i and dp[i][curr_c] != dp[i - 1][curr_c]:
            chosen_tasks.append(t)
            backtrack_chosen_indices.append(i)
            curr_c -= w_i

        backtrack_cells.append((i - 1, curr_c))

    chosen_tasks.reverse()
    total_weight = sum(t.weight for t in chosen_tasks)
    total_value = sum(t.value for t in chosen_tasks)
    cap_pct = round((total_weight / C) * 100.0, 1) if C > 0 else 0.0

    elapsed = (time.perf_counter() - start_time) * 1000.0

    return KnapsackResult(
        tasks=tasks,
        capacity=C,
        dp_table=dp,
        chosen_tasks=chosen_tasks,
        total_weight=total_weight,
        total_value=total_value,
        capacity_used_pct=cap_pct,
        backtrack_cells=backtrack_cells,
        backtrack_chosen_indices=backtrack_chosen_indices,
        steps=steps,
        time_elapsed_ms=elapsed,
    )


def solve_knapsack_bruteforce(tasks: List[Task], capacity: int) -> KnapsackResult:
    """
    Exhaustive O(2^n) brute-force solver for testing and verification against DP.
    """
    start_time = time.perf_counter()
    n = len(tasks)
    C = max(0, int(capacity))

    best_subset: List[Task] = []
    best_value = 0
    best_weight = 0

    # Test all 2^n binary combinations
    for k in range(n + 1):
        for combo in itertools.combinations(tasks, k):
            comb_weight = sum(t.weight for t in combo)
            if comb_weight <= C:
                comb_val = sum(t.value for t in combo)
                if comb_val > best_value:
                    best_value = comb_val
                    best_weight = comb_weight
                    best_subset = list(combo)

    cap_pct = round((best_weight / C) * 100.0, 1) if C > 0 else 0.0
    elapsed = (time.perf_counter() - start_time) * 1000.0

    return KnapsackResult(
        tasks=tasks,
        capacity=C,
        dp_table=[],
        chosen_tasks=best_subset,
        total_weight=best_weight,
        total_value=best_value,
        capacity_used_pct=cap_pct,
        backtrack_cells=[],
        backtrack_chosen_indices=[],
        steps=[],
        time_elapsed_ms=elapsed,
    )


# =====================================================================
# Graph Integration: Server Evaluation & Placement Recommendation
# =====================================================================

@dataclass
class ServerEvaluation:
    """Evaluation metrics for a candidate server."""
    server_id: int
    name: str
    capacity: int
    delay_ms: float
    is_eligible: bool
    path_from_entry: List[int]
    knapsack: Optional[KnapsackResult] = None
    total_value: int = 0
    total_weight: int = 0
    capacity_used_pct: float = 0.0
    num_tasks: int = 0


@dataclass
class PlacementEvaluationResult:
    """Summary of knapsack evaluations across all candidate servers in the network."""
    entry_node_id: int
    entry_node_name: str
    max_delay_ms: float
    tasks: List[Task]
    server_evaluations: Dict[int, ServerEvaluation]
    recommended_server_id: Optional[int]
    recommended_server_name: Optional[str]
    best_value: int
    best_delay_ms: float
    eligible_count: int
    total_servers: int
    status_message: str

    def summary_dataframe(self) -> pd.DataFrame:
        """Returns comparison table of all servers in the network."""
        records = []
        for sid, ev in self.server_evaluations.items():
            if math.isinf(ev.delay_ms):
                delay_str = "Undefined (-∞, Negative Cycle)" if ev.delay_ms < 0 else "Unreachable (∞)"
            else:
                delay_str = f"{ev.delay_ms:.1f}"
            records.append({
                "Server ID": sid,
                "Server Name": ev.name,
                "Capacity (Units)": ev.capacity,
                "Delay from Entry (ms)": delay_str,
                "Eligible (<= Max Delay)": "YES" if ev.is_eligible else "NO (Exceeds Delay)",
                "Total Value Achieved": ev.total_value if ev.is_eligible else "-",
                "Capacity Used": f"{ev.total_weight}/{ev.capacity} ({ev.capacity_used_pct}%)" if ev.is_eligible else "-",
                "Tasks Placed": f"{ev.num_tasks}/{len(self.tasks)}" if ev.is_eligible else "-",
                "Is Recommended": "★ RECOMMENDED" if sid == self.recommended_server_id else "",
            })
        return pd.DataFrame(records)


def evaluate_all_servers_placement(
    network: CloudNetwork,
    entry_node_id: int,
    tasks: List[Task],
    max_delay_ms: float,
    dijkstra_delays: Optional[Dict[int, float]] = None,
    allow_negative_weights: bool = False,
) -> PlacementEvaluationResult:
    """
    Evaluates 0/1 Knapsack on each server in the network:
    1. Computes single-source shortest path delays from entry_node_id via Dijkstra (or Bellman-Ford if negative weights are enabled).
    2. If a reachable negative cycle is detected from entry_node_id, gracefully blocks placement.
    3. Identifies ELIGIBLE servers where delay_ms <= max_delay_ms and delay_ms is finite.
    4. Runs 0/1 Knapsack DP on each eligible server using its physical capacity.
    5. Recommends the server that MAXIMIZES total task value (delay as tie-breaker).
    """
    if network.num_servers == 0:
        return PlacementEvaluationResult(
            entry_node_id=entry_node_id,
            entry_node_name="None",
            max_delay_ms=max_delay_ms,
            tasks=tasks,
            server_evaluations={},
            recommended_server_id=None,
            recommended_server_name=None,
            best_value=0,
            best_delay_ms=float("inf"),
            eligible_count=0,
            total_servers=0,
            status_message="Network contains 0 servers.",
        )

    if entry_node_id not in network.servers:
        entry_node_id = min(network.servers.keys())

    entry_server = network.servers[entry_node_id]

    # Precompute routing delays from user entry node
    # If negative weights are enabled or present, use Bellman-Ford
    use_bellman_ford = allow_negative_weights or network.has_negative_weights()

    if use_bellman_ford:
        bf_res = bellman_ford_shortest_path(network, source=entry_node_id)
        if bf_res.has_negative_cycle:
            # Placement blocked due to reachable negative cycle
            server_evals: Dict[int, ServerEvaluation] = {}
            for sid, srv in network.servers.items():
                d = bf_res.distances.get(sid, float("inf"))
                server_evals[sid] = ServerEvaluation(
                    server_id=sid,
                    name=srv.name,
                    capacity=srv.capacity,
                    delay_ms=d,
                    is_eligible=False,
                    path_from_entry=[],
                    knapsack=None,
                    total_value=0,
                    total_weight=0,
                    capacity_used_pct=0.0,
                    num_tasks=0,
                )
            cycle_desc = " → ".join(str(cid) for cid in bf_res.negative_cycle_nodes) if bf_res.negative_cycle_nodes else "Reachable cycle"
            return PlacementEvaluationResult(
                entry_node_id=entry_node_id,
                entry_node_name=entry_server.name,
                max_delay_ms=max_delay_ms,
                tasks=tasks,
                server_evaluations=server_evals,
                recommended_server_id=None,
                recommended_server_name=None,
                best_value=0,
                best_delay_ms=float("-inf"),
                eligible_count=0,
                total_servers=len(network.servers),
                status_message=f"Placement blocked: Reachable negative cycle detected from user entry node ({cycle_desc}). Shortest-path delays are undefined (-infinity).",
            )
        routing_delays = bf_res.distances
        routing_paths = {sid: bf_res.get_path_to(sid) for sid in network.servers}
    elif dijkstra_delays is not None:
        routing_delays = dijkstra_delays
        routing_paths = {sid: [entry_node_id, sid] if sid != entry_node_id else [entry_node_id] for sid in network.servers}
    else:
        dijk_res = dijkstra_shortest_path(network, source=entry_node_id)
        routing_delays = dijk_res.distances
        routing_paths = {sid: dijk_res.get_path_to(sid) for sid in network.servers}

    server_evals = {}
    eligible_servers: List[ServerEvaluation] = []

    for sid, srv in network.servers.items():
        delay = float(routing_delays.get(sid, float("inf")))
        is_eligible = (not math.isinf(delay)) and (delay <= max_delay_ms)
        path = routing_paths.get(sid, [sid])

        if is_eligible:
            knapsack_res = solve_knapsack_01_dp(tasks, capacity=srv.capacity)
            ev = ServerEvaluation(
                server_id=sid,
                name=srv.name,
                capacity=srv.capacity,
                delay_ms=round(delay, 2),
                is_eligible=True,
                path_from_entry=path,
                knapsack=knapsack_res,
                total_value=knapsack_res.total_value,
                total_weight=knapsack_res.total_weight,
                capacity_used_pct=knapsack_res.capacity_used_pct,
                num_tasks=len(knapsack_res.chosen_tasks),
            )
            eligible_servers.append(ev)
        else:
            ev = ServerEvaluation(
                server_id=sid,
                name=srv.name,
                capacity=srv.capacity,
                delay_ms=round(delay, 2) if not math.isinf(delay) else float("inf"),
                is_eligible=False,
                path_from_entry=path,
                knapsack=None,
                total_value=0,
                total_weight=0,
                capacity_used_pct=0.0,
                num_tasks=0,
            )

        server_evals[sid] = ev

    # Rank eligible servers:
    # Primary: Total Value (descending)
    # Tie-breaker 1: Delay from entry node (ascending)
    # Tie-breaker 2: Capacity used percentage (descending)
    if eligible_servers:
        eligible_servers.sort(key=lambda s: (-s.total_value, s.delay_ms, -s.capacity_used_pct))
        best_server = eligible_servers[0]
        rec_id = best_server.server_id
        rec_name = best_server.name
        best_val = best_server.total_value
        best_del = best_server.delay_ms
        msg = f"Recommended Server: '{rec_name}' (ID {rec_id}) achieves maximum value of {best_val} pts with {best_del:.1f} ms delay."
    else:
        rec_id = None
        rec_name = None
        best_val = 0
        best_del = float("inf")
        msg = f"No servers are eligible under the maximum allowed delay threshold of {max_delay_ms:.1f} ms."

    return PlacementEvaluationResult(
        entry_node_id=entry_node_id,
        entry_node_name=entry_server.name,
        max_delay_ms=max_delay_ms,
        tasks=tasks,
        server_evaluations=server_evals,
        recommended_server_id=rec_id,
        recommended_server_name=rec_name,
        best_value=best_val,
        best_delay_ms=best_del,
        eligible_count=len(eligible_servers),
        total_servers=len(network.servers),
        status_message=msg,
    )


# =====================================================================
# Backward-Compatibility Adapters
# =====================================================================

@dataclass
class PlacementResult:
    """Compatibility adapter mapping new KnapsackResult to legacy PlacementResult properties."""
    algorithm_name: str
    selected_server_ids: List[int]
    selected_server_names: List[str]
    entry_node_id: int
    entry_node_name: str
    target_demand: int
    total_capacity: int
    total_delay: float
    is_feasible: bool
    status_message: str
    server_details: List[Any] = field(default_factory=list)
    dp_table: Optional[Any] = None
    server_records: List[Dict[str, Any]] = field(default_factory=list)
    backtrack_path: List[Tuple[int, int]] = field(default_factory=list)
    subsets_evaluated: int = 0
    time_elapsed_ms: float = 0.0

    @property
    def host_server_ids(self) -> List[int]:
        return self.selected_server_ids

    @property
    def host_server_names(self) -> List[str]:
        return self.selected_server_names

    @property
    def avg_latency(self) -> float:
        return self.total_delay

    @property
    def max_latency(self) -> float:
        return self.total_delay

    def assignments_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame()

    def dp_dataframe(self, max_columns: int = 30) -> pd.DataFrame:
        return pd.DataFrame()


def solve_placement_dp(
    network: CloudNetwork,
    entry_node_id: int,
    demand: Optional[int] = None,
    k: Optional[int] = None,
    dijkstra_delays: Optional[Dict[int, float]] = None,
    tasks: Optional[List[Task]] = None,
    max_delay_ms: float = 100.0,
) -> PlacementResult:
    """
    Backward-compatibility adapter executing the 0/1 Knapsack placement evaluation.
    """
    if tasks is None:
        tasks = get_default_tasks()

    eval_res = evaluate_all_servers_placement(
        network=network,
        entry_node_id=entry_node_id,
        tasks=tasks,
        max_delay_ms=max_delay_ms,
        dijkstra_delays=dijkstra_delays,
    )

    if eval_res.recommended_server_id is not None:
        rec_id = eval_res.recommended_server_id
        ev = eval_res.server_evaluations[rec_id]
        return PlacementResult(
            algorithm_name="0/1 Knapsack DP (Max Value)",
            selected_server_ids=[rec_id],
            selected_server_names=[ev.name],
            entry_node_id=entry_node_id,
            entry_node_name=eval_res.entry_node_name,
            target_demand=ev.total_weight,
            total_capacity=ev.capacity,
            total_delay=ev.delay_ms,
            is_feasible=True,
            status_message=eval_res.status_message,
            dp_table=ev.knapsack.dp_table if ev.knapsack else None,
            server_records=[{"id": rec_id, "name": ev.name, "capacity": ev.capacity, "delay": ev.delay_ms}],
            backtrack_path=ev.knapsack.backtrack_cells if ev.knapsack else [],
            time_elapsed_ms=ev.knapsack.time_elapsed_ms if ev.knapsack else 0.0,
        )
    else:
        return PlacementResult(
            algorithm_name="0/1 Knapsack DP (Max Value)",
            selected_server_ids=[],
            selected_server_names=[],
            entry_node_id=entry_node_id,
            entry_node_name=eval_res.entry_node_name,
            target_demand=0,
            total_capacity=0,
            total_delay=float("inf"),
            is_feasible=False,
            status_message=eval_res.status_message,
        )
