"""
baselines.py - Heuristic Baselines for 0/1 Knapsack Task Placement Comparison.

Algorithms Implemented:
1. Dynamic Programming (0/1 Knapsack - Exact Optimum):
   - Computes global optimal subset of tasks maximizing total value under capacity constraint.
   - Time Complexity: O(n * C).

2. Greedy Heuristic (Highest Value First):
   - Sorts candidate tasks by absolute value in descending order.
   - Greedily includes tasks that fit within remaining capacity.
   - Time Complexity: O(n log n).

3. Greedy Heuristic (Value / Weight Density Ratio):
   - Sorts candidate tasks by efficiency ratio (value_i / weight_i) in descending order.
   - Greedily includes highest-density tasks that fit within remaining capacity.
   - Time Complexity: O(n log n).

4. Random Baseline (Monte Carlo Sampling):
   - Randomly samples subsets of tasks respecting capacity over multiple trials.
   - Retains the trial that achieves maximum total value.
   - Time Complexity: O(trials * n).
"""

from __future__ import annotations
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from placement_dp import Task, KnapsackResult, solve_knapsack_01_dp


@dataclass
class KnapsackBaselineResult:
    """Standardized output for knapsack heuristics."""
    algorithm_name: str
    chosen_tasks: List[Task]
    total_value: int
    total_weight: int
    capacity: int
    capacity_used_pct: float
    time_elapsed_ms: float
    strategy_description: str


def solve_knapsack_greedy_value(tasks: List[Task], capacity: int) -> KnapsackBaselineResult:
    """
    Greedy heuristic: Prioritizes tasks with the HIGHEST absolute value.
    Sorts tasks by value descending (ties broken by lower weight).
    """
    start_time = time.perf_counter()
    C = max(0, int(capacity))

    # Sort descending by value; tie-break by ascending weight
    sorted_tasks = sorted(tasks, key=lambda t: (-t.value, t.weight))

    chosen: List[Task] = []
    curr_weight = 0
    curr_val = 0

    for t in sorted_tasks:
        if curr_weight + t.weight <= C:
            chosen.append(t)
            curr_weight += t.weight
            curr_val += t.value

    cap_pct = round((curr_weight / C) * 100.0, 1) if C > 0 else 0.0
    elapsed = (time.perf_counter() - start_time) * 1000.0

    return KnapsackBaselineResult(
        algorithm_name="Greedy (Highest Value)",
        chosen_tasks=chosen,
        total_value=curr_val,
        total_weight=curr_weight,
        capacity=C,
        capacity_used_pct=cap_pct,
        time_elapsed_ms=elapsed,
        strategy_description="Sorts tasks by value descending; packs heaviest-value tasks first.",
    )


def solve_knapsack_greedy_ratio(tasks: List[Task], capacity: int) -> KnapsackBaselineResult:
    """
    Greedy heuristic: Prioritizes tasks with the HIGHEST value-to-weight density ratio.
    Sorts tasks by (value / weight) descending (ties broken by higher value).
    """
    start_time = time.perf_counter()
    C = max(0, int(capacity))

    # Sort descending by ratio; tie-break by descending value
    sorted_tasks = sorted(
        tasks,
        key=lambda t: (-(t.value / t.weight) if t.weight > 0 else -float("inf"), -t.value)
    )

    chosen: List[Task] = []
    curr_weight = 0
    curr_val = 0

    for t in sorted_tasks:
        if curr_weight + t.weight <= C:
            chosen.append(t)
            curr_weight += t.weight
            curr_val += t.value

    cap_pct = round((curr_weight / C) * 100.0, 1) if C > 0 else 0.0
    elapsed = (time.perf_counter() - start_time) * 1000.0

    return KnapsackBaselineResult(
        algorithm_name="Greedy (Value / Weight Ratio)",
        chosen_tasks=chosen,
        total_value=curr_val,
        total_weight=curr_weight,
        capacity=C,
        capacity_used_pct=cap_pct,
        time_elapsed_ms=elapsed,
        strategy_description="Sorts tasks by value-to-weight ratio descending; packs highest-density tasks first.",
    )


def solve_knapsack_random(
    tasks: List[Task],
    capacity: int,
    seed: Optional[int] = 42,
    max_trials: int = 50,
) -> KnapsackBaselineResult:
    """
    Random baseline: Shuffles candidate tasks randomly and packs until capacity is reached.
    Evaluates max_trials permutations and keeps the trial yielding highest total value.
    """
    start_time = time.perf_counter()
    rng = random.Random(seed)
    C = max(0, int(capacity))

    best_subset: List[Task] = []
    best_val = 0
    best_weight = 0

    for _ in range(max_trials):
        shuffled = tasks[:]
        rng.shuffle(shuffled)

        trial_chosen: List[Task] = []
        trial_w = 0
        trial_v = 0

        for t in shuffled:
            if trial_w + t.weight <= C:
                trial_chosen.append(t)
                trial_w += t.weight
                trial_v += t.value

        if trial_v > best_val:
            best_val = trial_v
            best_weight = trial_w
            best_subset = trial_chosen

    cap_pct = round((best_weight / C) * 100.0, 1) if C > 0 else 0.0
    elapsed = (time.perf_counter() - start_time) * 1000.0

    return KnapsackBaselineResult(
        algorithm_name="Random Baseline",
        chosen_tasks=best_subset,
        total_value=best_val,
        total_weight=best_weight,
        capacity=C,
        capacity_used_pct=cap_pct,
        time_elapsed_ms=elapsed,
        strategy_description=f"Samples {max_trials} random packing permutations; retains maximum value trial.",
    )


def run_knapsack_comparison(
    tasks: List[Task],
    capacity: int,
    num_random_trials: int = 60,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Compares Dynamic Programming vs Greedy (Value) vs Greedy (Ratio) vs Random
    on total task value, total weight, capacity utilization, and runtime.

    Returns:
        results_map: Dict mapping algorithm names to their result dataclass
        summary_df: Formatted pandas DataFrame for display
    """
    results: Dict[str, Any] = {}

    # 1. Exact 0/1 Knapsack DP
    t0 = time.perf_counter()
    dp_res = solve_knapsack_01_dp(tasks, capacity)
    t1 = time.perf_counter()
    dp_res.time_elapsed_ms = round((t1 - t0) * 1000.0, 2)
    results["0/1 Knapsack DP (Optimal)"] = dp_res

    # 2. Greedy (Highest Value)
    t0 = time.perf_counter()
    g_val_res = solve_knapsack_greedy_value(tasks, capacity)
    t1 = time.perf_counter()
    g_val_res.time_elapsed_ms = round((t1 - t0) * 1000.0, 2)
    results["Greedy (Highest Value)"] = g_val_res

    # 3. Greedy (Value/Weight Ratio)
    t0 = time.perf_counter()
    g_rat_res = solve_knapsack_greedy_ratio(tasks, capacity)
    t1 = time.perf_counter()
    g_rat_res.time_elapsed_ms = round((t1 - t0) * 1000.0, 2)
    results["Greedy (Value / Weight Ratio)"] = g_rat_res

    # 4. Random Baseline
    t0 = time.perf_counter()
    rand_res = solve_knapsack_random(tasks, capacity, seed=42, max_trials=num_random_trials)
    t1 = time.perf_counter()
    rand_res.time_elapsed_ms = round((t1 - t0) * 1000.0, 2)
    results["Random Baseline"] = rand_res

    opt_val = dp_res.total_value

    records = []
    for algo_name, res in results.items():
        val = res.total_value
        weight = res.total_weight
        cap_pct = res.capacity_used_pct
        runtime = res.time_elapsed_ms
        task_names = [t.name for t in res.chosen_tasks]

        if opt_val > 0:
            if val == opt_val:
                gap_str = "0.0% (Optimal)"
            else:
                gap_pct = ((opt_val - val) / opt_val) * 100.0
                gap_str = f"-{gap_pct:.1f}%"
        else:
            gap_str = "0.0%"

        records.append({
            "Algorithm": algo_name,
            "Total Value (Points)": val,
            "Total Weight (Units)": f"{weight} / {capacity}",
            "Capacity Used (%)": f"{cap_pct}%",
            "Tasks Chosen": ", ".join(task_names) if task_names else "None",
            "Count": len(res.chosen_tasks),
            "Gap vs DP": gap_str,
            "Runtime (ms)": f"{runtime:.2f}",
        })

    summary_df = pd.DataFrame(records)
    return results, summary_df


# =====================================================================
# Backward Compatibility Placeholders
# =====================================================================

def solve_placement_greedy(*args, **kwargs):
    tasks = kwargs.get("tasks") or (args[2] if len(args) > 2 and isinstance(args[2], list) else None)
    cap = kwargs.get("demand") or 50
    if tasks:
        return solve_knapsack_greedy_ratio(tasks, cap)
    from placement_dp import get_default_tasks
    return solve_knapsack_greedy_ratio(get_default_tasks(), 50)


def solve_placement_random(*args, **kwargs):
    from placement_dp import get_default_tasks
    return solve_knapsack_random(get_default_tasks(), 50)


def run_benchmark_comparison(*args, **kwargs):
    from placement_dp import get_default_tasks
    return run_knapsack_comparison(get_default_tasks(), 50)
