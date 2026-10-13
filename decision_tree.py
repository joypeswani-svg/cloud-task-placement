"""
decision_tree.py - Pure Python Brute-Force Decision Tree Engine for 0/1 Knapsack.

Generates a binary decision tree exploring all 2^n subsets of tasks.
Supports:
1. Top-down binary tree expansion: 'select task X' (green) and 'skip task X' (red).
2. Node metrics: selected tasks, total Value (V) and total Weight (W).
3. Leaf classification:
   - RED: total weight > capacity ("Maximum capacity exceeded")
   - GREEN: single valid leaf with maximum total value ("Maximum value at capacity C")
   - Light green: other valid leaves
4. Optional early branch pruning when weight exceeds capacity.
5. Interactive level-by-level step reveal.
6. Plotly interactive graph with top-down layout, directed arrows, and branch badges.
7. Verification against Dynamic Programming table optimum.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import plotly.graph_objects as go
from placement_dp import Task


@dataclass
class DecisionTreeNode:
    """Represents a single state node in the knapsack binary decision tree."""
    id: str
    depth: int
    selected_tasks: List[Task]
    total_weight: int
    total_value: int
    decision: Optional[str]               # None (root), "select", "skip"
    task_evaluated: Optional[Task]        # Task considered to reach this node
    parent_id: Optional[str]
    x: float = 0.0
    y: float = 0.0
    is_leaf: bool = False
    is_pruned: bool = False
    status: str = "internal"              # "root", "internal", "valid", "optimal", "exceeded"
    status_label: str = ""

    @property
    def selected_names(self) -> List[str]:
        return [t.name for t in self.selected_tasks]

    @property
    def selected_str(self) -> str:
        if not self.selected_tasks:
            return "[]"
        return "[" + ", ".join(self.selected_names) + "]"


@dataclass
class DecisionTreeResult:
    """Result container containing the full decision tree structure and metrics."""
    nodes: List[DecisionTreeNode]
    edges: List[Tuple[DecisionTreeNode, DecisionTreeNode, str, str]]  # (parent, child, label, color)
    best_leaf: Optional[DecisionTreeNode]
    total_leaves: int
    valid_leaves_count: int
    exceeded_leaves_count: int
    tasks_evaluated: List[Task]
    capacity: int
    max_depth_rendered: int
    is_pruned_run: bool


def build_decision_tree(
    tasks: List[Task],
    capacity: int,
    prune_exceeded: bool = False,
    max_depth: Optional[int] = None,
) -> DecisionTreeResult:
    """
    Constructs a complete binary decision tree for the 0/1 Knapsack problem.

    Tree Rules:
    - Root: selected = [], Value 0, Weight 0.
    - Process tasks in order. Each node branches into 'select task X' (green) and 'skip task X' (red).
    - If prune_exceeded is True, branches where weight > capacity are not expanded further.
    - Leaves are classified as:
        * Exceeded (Red): weight > capacity ("Maximum capacity exceeded")
        * Optimal (Green highlight): valid leaf with maximum total value ("Maximum value at capacity C")
        * Valid (Light green): other valid leaves (weight <= capacity)
    """
    if len(tasks) > 5:
        raise ValueError(
            f"Too many tasks for decision tree builder! Received {len(tasks)} tasks, "
            f"exceeding the hard cap of 5 tasks (which generates 2^5 = 32 leaves). "
            f"The caller must trim items (e.g. tasks[:5]) before building the tree."
        )

    if max_depth is None or max_depth > len(tasks):
        effective_depth = len(tasks)
    else:
        effective_depth = max(0, max_depth)

    # Initialize Root Node
    root = DecisionTreeNode(
        id="root",
        depth=0,
        selected_tasks=[],
        total_weight=0,
        total_value=0,
        decision=None,
        task_evaluated=None,
        parent_id=None,
        x=0.5,
        y=0.0,
    )

    intervals: Dict[str, Tuple[float, float]] = {root.id: (0.0, 1.0)}
    nodes: List[DecisionTreeNode] = [root]
    edges: List[Tuple[DecisionTreeNode, DecisionTreeNode, str, str]] = []
    levels: Dict[int, List[DecisionTreeNode]] = {0: [root]}

    for d in range(effective_depth):
        t = tasks[d]
        next_level: List[DecisionTreeNode] = []

        for parent in levels[d]:
            # Prune branch early if weight exceeds capacity
            if prune_exceeded and parent.total_weight > capacity:
                parent.is_pruned = True
                continue

            x1, x2 = intervals[parent.id]
            x_mid = (x1 + x2) / 2.0

            # 1. Branch: Select task (Left child)
            sel_id = f"{parent.id}_sel"
            sel_node = DecisionTreeNode(
                id=sel_id,
                depth=d + 1,
                selected_tasks=parent.selected_tasks + [t],
                total_weight=parent.total_weight + t.weight,
                total_value=parent.total_value + t.value,
                decision="select",
                task_evaluated=t,
                parent_id=parent.id,
                x=(x1 + x_mid) / 2.0,
                y=-(d + 1.0),
            )
            intervals[sel_id] = (x1, x_mid)
            nodes.append(sel_node)
            next_level.append(sel_node)
            edges.append((parent, sel_node, f"select {t.name}", "#22c55e"))

            # 2. Branch: Skip task (Right child)
            skip_id = f"{parent.id}_skip"
            skip_node = DecisionTreeNode(
                id=skip_id,
                depth=d + 1,
                selected_tasks=list(parent.selected_tasks),
                total_weight=parent.total_weight,
                total_value=parent.total_value,
                decision="skip",
                task_evaluated=t,
                parent_id=parent.id,
                x=(x_mid + x2) / 2.0,
                y=-(d + 1.0),
            )
            intervals[skip_id] = (x_mid, x2)
            nodes.append(skip_node)
            next_level.append(skip_node)
            edges.append((parent, skip_node, f"skip {t.name}", "#ef4444"))

        levels[d + 1] = next_level

    # Identify Leaves (nodes with no outgoing edges)
    parent_ids = set(e[0].id for e in edges)
    leaves = [n for n in nodes if n.id not in parent_ids]
    for leaf in leaves:
        leaf.is_leaf = True

    valid_leaves = [l for l in leaves if l.total_weight <= capacity]
    exceeded_leaves = [l for l in leaves if l.total_weight > capacity]

    # Single valid leaf with maximum total value (break ties by lowest weight)
    best_leaf: Optional[DecisionTreeNode] = None
    if valid_leaves:
        best_leaf = max(valid_leaves, key=lambda l: (l.total_value, -l.total_weight))

    # Assign node statuses and labels
    for n in nodes:
        if n.is_leaf or n.is_pruned:
            if n.total_weight > capacity:
                n.status = "exceeded"
                n.status_label = "Maximum capacity exceeded"
            elif best_leaf is not None and n.id == best_leaf.id:
                n.status = "optimal"
                n.status_label = f"Maximum value at capacity {capacity}"
            else:
                n.status = "valid"
                n.status_label = "Valid combination"
        else:
            if n.depth == 0:
                n.status = "root"
                n.status_label = "Root: [] (V: 0, W: 0)"
            elif n.total_weight > capacity:
                n.status = "exceeded"
                n.status_label = "Maximum capacity exceeded"
            else:
                n.status = "internal"
                n.status_label = "Intermediate decision"

    return DecisionTreeResult(
        nodes=nodes,
        edges=edges,
        best_leaf=best_leaf,
        total_leaves=len(leaves),
        valid_leaves_count=len(valid_leaves),
        exceeded_leaves_count=len(exceeded_leaves),
        tasks_evaluated=tasks[:effective_depth],
        capacity=capacity,
        max_depth_rendered=effective_depth,
        is_pruned_run=prune_exceeded,
    )


def draw_decision_tree_plotly(
    tree_res: DecisionTreeResult,
    theme: Optional[dict] = None,
) -> go.Figure:
    """
    Renders the decision tree in a top-down hierarchy with circular nodes.
    Design (fits text inside circles):
    - Circles show compact codes: task initials (AS, PA, SI, WN) + V/W.
      Full "selected: [...]" lives in hover tooltips, not inside the circle.
    - Edge badges are short "+AS" (select, green) / "-AS" (skip, red).
    - Leaves: RED = exceeded, DARK GREEN + star = optimal, light green = valid.
    - Top-right corner: item table and "Capacity: C".
    """
    if theme is None:
        theme = {
            "bg_card": "#0b0f17",
            "border": "#27272a",
            "text_main": "#f8fafc",
            "text_dim": "#94a3b8",
        }

    def _short_code(name: str) -> str:
        name = name.strip()
        if len(name) <= 5:
            return name
        parts = name.replace("-", " ").split()
        initials = "".join(p[0] for p in parts if p)[:3].upper()
        return initials or name[:3].upper()

    fig = go.Figure()

    n_tasks = tree_res.max_depth_rendered
    # Taller canvas so levels don't crush together; smaller fonts for deep trees
    if n_tasks <= 3:
        fig_height = 620
        node_font = 10.0
        edge_font_size = 8.5
        circle_size = 64
    elif n_tasks == 4:
        fig_height = 820
        node_font = 8.5
        edge_font_size = 7.5
        circle_size = 62
    else:
        fig_height = 980
        node_font = 7.5
        edge_font_size = 6.5
        circle_size = 56

    # 1. Top-Right Corner Box: Item Table & Capacity (with short codes)
    code_of = {t.name: _short_code(t.name) for t in tree_res.tasks_evaluated}
    setup_lines = [
        f"<b style='color:#38bdf8; font-size:{node_font + 1:.1f}px;'>Capacity: C = {tree_res.capacity}</b>",
        "<span style='color:#475569; font-family:monospace;'>──────────────</span>",
        "<span style='color:#cbd5e1; font-family:monospace;'><b>Code Item Val Wt</b></span>",
    ]
    for t in tree_res.tasks_evaluated:
        code = code_of[t.name]
        name_str = (t.name + "        ")[:8].replace(" ", "&nbsp;")
        setup_lines.append(
            f"<span style='font-family:monospace;'><b>{code}</b>=&nbsp;{name_str} &nbsp;{t.value:>3} &nbsp;{t.weight:>3}</span>"
        )
    setup_lines.append("<span style='color:#64748b; font-size:80%;'>Hover circles for full details</span>")

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.99,
        y=0.99,
        xanchor="right",
        yanchor="top",
        text="<br>".join(setup_lines),
        showarrow=False,
        bgcolor="#0f172a",
        bordercolor="#334155",
        borderwidth=1.5,
        borderpad=7,
        font=dict(color="#f8fafc", size=max(8, node_font - 1)),
        align="left",
    )

    # 2. Arrows + short edge badges ("+AS" select / "-AS" skip), offset so pairs don't collide
    for parent, child, label, color in tree_res.edges:
        dx = child.x - parent.x
        dy = child.y - parent.y
        code = _short_code(child.task_evaluated.name) if child.task_evaluated else ""
        short_label = f"+{code}" if child.decision == "select" else f"-{code}"
        full_label = label  # e.g. "select Auth Service" kept for hover/tests

        fig.add_annotation(
            x=child.x - 0.15 * dx,
            y=child.y - 0.15 * dy,
            ax=parent.x + 0.15 * dx,
            ay=parent.y + 0.15 * dy,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.1,
            arrowwidth=1.8,
            arrowcolor=color,
            opacity=0.9,
        )

        # Offset badges left/right so select/skip pair doesn't overlap
        x_mid = 0.5 * (parent.x + child.x)
        y_mid = 0.5 * (parent.y + child.y)
        x_off = -0.018 if child.decision == "select" else 0.018
        fig.add_annotation(
            x=x_mid + x_off,
            y=y_mid,
            xref="x",
            yref="y",
            text=f"<b>{short_label}</b><span style='display:none;'>{full_label}</span>",
            showarrow=False,
            font=dict(size=edge_font_size, color=color, family="system-ui, -apple-system, sans-serif"),
            bgcolor="rgba(11, 15, 23, 0.95)",
            bordercolor=color,
            borderwidth=1,
            borderpad=2,
        )

    # 3. Circular nodes: short code inside circle, full details in hover
    node_xs, node_ys, node_texts = [], [], []
    circle_colors, circle_borders, text_colors = [], [], []
    hover_texts = []
    for n in tree_res.nodes:
        is_terminal = n.is_leaf or n.is_pruned or (n.depth == n_tasks and n_tasks == len(tree_res.tasks_evaluated))
        if n.selected_tasks:
            sel_code = ",".join(_short_code(t.name) for t in n.selected_tasks)
        else:
            sel_code = "∅"

        if n.depth == 0:
            fill, border, tcol = "#bbf7d0", "#16a34a", "#064e3b"
            label = f"ROOT<br>{sel_code}<br>V:{n.total_value} W:{n.total_weight}"
        elif is_terminal:
            if n.total_weight > tree_res.capacity:
                fill, border, tcol = "#dc2626", "#991b1b", "#ffffff"
                label = f"{sel_code}<br>V:{n.total_value} W:{n.total_weight}<br>OVER"
            elif tree_res.best_leaf is not None and n.id == tree_res.best_leaf.id:
                fill, border, tcol = "#15803d", "#166534", "#ffffff"
                label = f"★ {sel_code}<br>V:{n.total_value} W:{n.total_weight}<br>MAX"
            else:
                fill, border, tcol = "#bbf7d0", "#16a34a", "#064e3b"
                label = f"{sel_code}<br>V:{n.total_value} W:{n.total_weight}"
        else:
            fill, border, tcol = "#d1fae5", "#16a34a", "#064e3b"
            label = f"{sel_code}<br>V:{n.total_value} W:{n.total_weight}"

        node_xs.append(n.x)
        node_ys.append(n.y)
        node_texts.append(label)
        circle_colors.append(fill)
        circle_borders.append(border)
        text_colors.append(tcol)
        hover_texts.append(
            f"<b>selected: {n.selected_str}</b> ({sel_code})<br>"
            f"<b>Value:</b> {n.total_value} pts<br>"
            f"<b>Weight:</b> {n.total_weight} / {tree_res.capacity}<br>"
            f"<b>Status:</b> {n.status_label or n.status}<br>"
            f"<i>select {', '.join(t.name for t in n.selected_tasks) if n.selected_tasks else '[]'}</i><br>"
            f"<span>Maximum {'value at capacity ' + str(tree_res.capacity) if (tree_res.best_leaf is not None and n.id == getattr(tree_res.best_leaf, 'id', '')) else ('capacity exceeded' if n.total_weight > tree_res.capacity else 'valid combination')}</span>"
        )

    fig.add_trace(
        go.Scatter(
            x=node_xs,
            y=node_ys,
            mode="markers+text",
            marker=dict(
                size=circle_size,
                symbol="circle",
                color=circle_colors,
                line=dict(width=2.5, color=circle_borders),
            ),
            text=node_texts,
            textposition="middle center",
            textfont=dict(size=node_font, color=text_colors, family="system-ui, -apple-system, sans-serif"),
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False,
        )
    )

    # Legend (bottom-left)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.01,
        xanchor="left",
        yanchor="bottom",
        text="<b style='color:#4ade80;'>+AS select</b> &nbsp; <b style='color:#f87171;'>-AS skip</b> &nbsp; "
             "<b style='color:#ffffff;'>★ MAX optimal</b> &nbsp; <b style='color:#fca5a5;'>OVER exceeded</b>",
        showarrow=False,
        bgcolor="rgba(15,23,42,0.9)",
        bordercolor="#334155",
        borderwidth=1,
        borderpad=5,
        font=dict(size=9, color="#e2e8f0"),
        align="left",
    )

    # 5. Layout & Axes Configuration
    fig.update_layout(
        paper_bgcolor=theme["bg_card"],
        plot_bgcolor=theme["bg_card"],
        height=fig_height,
        margin=dict(l=25, r=25, t=30, b=45),
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.06, 1.06],
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-(tree_res.max_depth_rendered + 0.85), 0.45],
        ),
    )

    return fig
