"""
visualization.py - Plotly Dark Theme Visualizations for Cloud Task Placement Simulator.

Uses shades of black, charcoal, and obsidian with neon accents for high-contrast,
interactive college presentation-grade visualizations.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional, Any, Set
import plotly.graph_objects as go
import pandas as pd
from graph_utils import CloudNetwork, Server, Link, Client


# Color Palette: Shades of Black & Obsidian with Very Dark Red Accents
THEME = {
    "bg_pitch_black": "#050608",
    "bg_charcoal": "#0d0f14",
    "bg_card": "#131720",
    "border_subtle": "#222938",
    "text_bright": "#f8fafc",
    "text_muted": "#94a3b8",
    "node_normal": "#1c1f26",
    "node_normal_border": "#3f3f46",
    "node_selected": "#8b0000",         # Very Dark Red
    "node_selected_border": "#dc2626",  # Crimson border
    "node_selected_glow": "rgba(139, 0, 0, 0.45)",
    "node_client": "#d97706",          # Amber
    "link_normal": "#27272a",          # Faint Charcoal
    "link_mst": "#991b1b",             # Very Dark Red / Deep Crimson
    "link_route": "#dc2626",           # Crimson Scarlet
    "link_faint": "#141416",           # Dimmed inactive link
}


def get_link_geometry(
    u: int,
    v: int,
    directed: bool,
    network: CloudNetwork,
    curve_offset: float = 40.0,
) -> Tuple[List[float], List[float], float, float, float, float, float, float]:
    """
    Computes 2D rendering geometry for a link from u to v.
    If multiple links or opposite directed links exist between u and v,
    returns quadratic bezier curve points so lines bow symmetrically in opposite directions.
    Returns:
      xs, ys: line coordinates
      arrow_x, arrow_y: point on line for directional arrow
      arrow_ux, arrow_uy: normalized direction vector for arrow
      label_x, label_y: coordinate for the latency badge
    """
    s_u = network.servers[u]
    s_v = network.servers[v]

    dx = s_v.x - s_u.x
    dy = s_v.y - s_u.y
    dist = math.hypot(dx, dy)
    if dist < 1e-4:
        dist = 1.0

    pair_links = [l for l in network.links if (l.u == u and l.v == v) or (l.u == v and l.v == u)]
    has_opposite = any(l.u == v and l.v == u for l in network.links)
    is_multi = len(pair_links) > 1

    # Unit normal vector pointing to the right of direction u -> v
    nx = -dy / dist
    ny = dx / dist

    mid_x = (s_u.x + s_v.x) / 2.0
    mid_y = (s_u.y + s_v.y) / 2.0

    if is_multi or has_opposite:
        same_dir = [l for l in pair_links if l.u == u and l.v == v]
        if has_opposite:
            eff_offset = curve_offset
        else:
            idx = same_dir.index(next(l for l in same_dir if l.u == u and l.v == v)) if same_dir else 0
            eff_offset = curve_offset * (1.0 if idx % 2 == 0 else -1.0)

        cx = mid_x + nx * eff_offset
        cy = mid_y + ny * eff_offset

        # 25 points along quadratic bezier curve
        xs = []
        ys = []
        for i in range(25):
            t = i / 24.0
            bx = (1.0 - t) ** 2 * s_u.x + 2.0 * (1.0 - t) * t * cx + t ** 2 * s_v.x
            by = (1.0 - t) ** 2 * s_u.y + 2.0 * (1.0 - t) * t * cy + t ** 2 * s_v.y
            xs.append(bx)
            ys.append(by)

        # Arrow positioned at t = 0.62 along curve
        t_arr = 0.62
        arrow_x = (1.0 - t_arr) ** 2 * s_u.x + 2.0 * (1.0 - t_arr) * t_arr * cx + t_arr ** 2 * s_v.x
        arrow_y = (1.0 - t_arr) ** 2 * s_u.y + 2.0 * (1.0 - t_arr) * t_arr * cy + t_arr ** 2 * s_v.y

        # Tangent vector at t_arr
        tan_x = 2.0 * (1.0 - t_arr) * (cx - s_u.x) + 2.0 * t_arr * (s_v.x - cx)
        tan_y = 2.0 * (1.0 - t_arr) * (cy - s_u.y) + 2.0 * t_arr * (s_v.y - cy)
        tan_len = math.hypot(tan_x, tan_y) or 1.0
        arrow_ux = tan_x / tan_len
        arrow_uy = tan_y / tan_len

        # Label badge at curve apex (t = 0.5) pushed slightly outward along normal
        apex_x = 0.25 * s_u.x + 0.5 * cx + 0.25 * s_v.x
        apex_y = 0.25 * s_u.y + 0.5 * cy + 0.25 * s_v.y
        label_x = apex_x + nx * 14.0
        label_y = apex_y + ny * 14.0

    else:
        # Straight line
        xs = [s_u.x, s_v.x]
        ys = [s_u.y, s_v.y]

        arrow_ux = dx / dist
        arrow_uy = dy / dist
        arrow_x = s_u.x + 0.62 * dx
        arrow_y = s_u.y + 0.62 * dy

        # Label offset perpendicular to line by 14px so it never collides with wire or arrow
        label_x = mid_x + nx * 14.0
        label_y = mid_y + ny * 14.0

    return xs, ys, arrow_x, arrow_y, arrow_ux, arrow_uy, label_x, label_y


def draw_network_graph(
    network: CloudNetwork,
    highlight_edges: Optional[Set[Tuple[int, int]]] = None,
    highlight_edge_color: str = THEME["link_mst"],
    highlight_nodes: Optional[Set[int]] = None,
    highlight_node_color: str = THEME["node_selected"],
    title: str = "Cloud Server Topology",
    route_edges: Optional[List[Tuple[int, int]]] = None,
    client_assignments: Optional[Dict[int, int]] = None,
    show_edge_labels: bool = True,
    height: int = 560,
    evaluating_edge: Optional[Tuple[int, int]] = None,
    evaluating_status: Optional[str] = "evaluating",  # "evaluating", "accepted", "rejected"
    rejected_edges: Optional[Set[Tuple[int, int]]] = None,
    negative_cycle_nodes: Optional[List[int]] = None,
) -> go.Figure:
    """
    Renders an interactive, dark-themed 2D cloud network topology with Plotly.
    Supports node scaling by capacity, edge highlighting for MST or shortest path,
    curved opposing directed links to eliminate visual overlap, negative weight rendering,
    negative cycle visualization, animated edge evaluation, and VM host server badges.
    """
    fig = go.Figure()

    highlight_edges = highlight_edges or set()
    highlight_nodes = highlight_nodes or set()
    route_set = set(route_edges or [])
    rejected_edges = rejected_edges or set()
    eval_key = (min(evaluating_edge[0], evaluating_edge[1]), max(evaluating_edge[0], evaluating_edge[1])) if evaluating_edge else None

    # Set of directed edges in negative cycle for overlay priority
    neg_cycle_edges: Set[Tuple[int, int]] = set()
    if negative_cycle_nodes and len(negative_cycle_nodes) >= 2:
        for idx in range(len(negative_cycle_nodes) - 1):
            neg_cycle_edges.add((negative_cycle_nodes[idx], negative_cycle_nodes[idx + 1]))

    # Map clients to servers for tooltip display
    server_clients: Dict[int, List[Client]] = {sid: [] for sid in network.servers}
    for c in network.clients:
        if c.server_id in server_clients:
            server_clients[c.server_id].append(c)

    # 1. Draw Links
    unique_links = network.unique_edges()

    # Normal background links (respecting directed vs undirected)
    for link in network.links:
        u, v, lat = link.u, link.v, link.latency
        if u not in network.servers or v not in network.servers:
            continue
        edge_key = (min(u, v), max(u, v))
        is_mst = edge_key in highlight_edges
        is_route = (u, v) in route_set or (not link.directed and (edge_key in route_set or (v, u) in route_set))
        is_eval = (eval_key == edge_key)
        is_rej = edge_key in rejected_edges
        is_neg_cycle = (u, v) in neg_cycle_edges or (not link.directed and (v, u) in neg_cycle_edges)

        # Skip drawing here if it will be drawn as highlighted/eval/route/neg_cycle overlay
        if is_mst or is_route or is_eval or is_rej or is_neg_cycle:
            continue

        s_u = network.servers[u]
        s_v = network.servers[v]

        link_color = "#f87171" if lat < 0 else THEME["link_normal"]
        dash_style = "dash" if lat < 0 else "solid"

        xs, ys, arr_x, arr_y, arr_ux, arr_uy, _, _ = get_link_geometry(u, v, link.directed, network)

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=link_color, width=2.2 if link.directed else 1.8, dash=dash_style),
                hoverinfo="text",
                hovertext=f"Link: {s_u.name} {'→' if link.directed else '↔'} {s_v.name}<br>Latency: {lat:.1f} ms",
                showlegend=False,
            )
        )

        # Midpoint directional arrow for directed links
        if link.directed:
            fig.add_annotation(
                x=arr_x + 6.0 * arr_ux,
                y=arr_y + 6.0 * arr_uy,
                ax=arr_x - 6.0 * arr_ux,
                ay=arr_y - 6.0 * arr_uy,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=2.0,
                arrowcolor=link_color,
            )

    # Rejected Cycle Edges (Faint dashed crimson)
    if rejected_edges:
        for u, v, lat in unique_links:
            edge_key = (min(u, v), max(u, v))
            if edge_key in rejected_edges and edge_key != eval_key and edge_key not in highlight_edges:
                s_u = network.servers[u]
                s_v = network.servers[v]
                xs, ys, _, _, _, _, _, _ = get_link_geometry(u, v, False, network)
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color="#ef4444", width=2.0, dash="dot"),
                        hoverinfo="text",
                        hovertext=f"✖ Rejected Cycle Link: {s_u.name} ↔ {s_v.name}<br>Latency: {lat:.1f} ms",
                        showlegend=False,
                    )
                )

    # MST Edges Overlay (Bold Cyan / Green)
    if highlight_edges:
        for u, v, lat in unique_links:
            edge_key = (min(u, v), max(u, v))
            if edge_key in highlight_edges and edge_key not in route_set and edge_key != eval_key:
                s_u = network.servers[u]
                s_v = network.servers[v]
                xs, ys, _, _, _, _, _, _ = get_link_geometry(u, v, False, network)
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color=highlight_edge_color, width=4.5),
                        hoverinfo="text",
                        hovertext=f"★ MST Backbone Link: {s_u.name} ↔ {s_v.name}<br>Latency: {lat:.1f} ms",
                        showlegend=False,
                    )
                )

    # Live Evaluating Edge Overlay (Pulsing Amber or Red/Green depending on evaluating_status)
    if eval_key is not None:
        u_eval, v_eval = eval_key
        if u_eval in network.servers and v_eval in network.servers:
            s_u = network.servers[u_eval]
            s_v = network.servers[v_eval]
            lat = next((l.latency for l in network.links if (l.u == u_eval and l.v == v_eval) or (l.u == v_eval and l.v == u_eval)), 0.0)
            is_dir = any(l.u == u_eval and l.v == v_eval and l.directed for l in network.links)

            if evaluating_status == "rejected":
                eval_color = "#ef4444"  # Bright Red
                eval_label = "✖ REJECTED (CYCLE DETECTED)"
            elif evaluating_status == "accepted":
                eval_color = "#22c55e"  # Vivid Green
                eval_label = "✔ ACCEPTED INTO MST"
            else:
                eval_color = "#f59e0b"  # Vibrant Amber
                eval_label = "⚡ CURRENTLY TESTING LINK"

            xs, ys, arr_x, arr_y, arr_ux, arr_uy, _, _ = get_link_geometry(u_eval, v_eval, is_dir, network)

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=eval_color, width=6.5),
                    hoverinfo="text",
                    hovertext=f"<b>{eval_label}</b><br>{s_u.name} ↔ {s_v.name}<br>Latency: {lat:.1f} ms",
                    showlegend=False,
                )
            )
            if is_dir:
                fig.add_annotation(
                    x=arr_x + 6.0 * arr_ux,
                    y=arr_y + 6.0 * arr_uy,
                    ax=arr_x - 6.0 * arr_ux,
                    ay=arr_y - 6.0 * arr_uy,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.3,
                    arrowwidth=2.5,
                    arrowcolor=eval_color,
                )

    # Active Route Overlay (Bold Neon Crimson)
    if route_edges:
        for u, v in route_edges:
            if u in network.servers and v in network.servers:
                s_u = network.servers[u]
                s_v = network.servers[v]
                lat = next((l.latency for l in network.links if (l.u == u and l.v == v) or (l.u == v and l.v == u)), 0.0)
                is_dir = any(l.u == u and l.v == v and l.directed for l in network.links)
                xs, ys, arr_x, arr_y, arr_ux, arr_uy, _, _ = get_link_geometry(u, v, is_dir, network)

                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color=THEME["link_route"], width=5.5),
                        hoverinfo="text",
                        hovertext=f"► Active Route Hop: {s_u.name} → {s_v.name}<br>Latency: {lat:.1f} ms",
                        showlegend=False,
                    )
                )
                if is_dir:
                    fig.add_annotation(
                        x=arr_x + 6.0 * arr_ux,
                        y=arr_y + 6.0 * arr_uy,
                        ax=arr_x - 6.0 * arr_ux,
                        ay=arr_y - 6.0 * arr_uy,
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.3,
                        arrowwidth=2.5,
                        arrowcolor=THEME["link_route"],
                    )

    # Negative Cycle Highlight Overlay (Curved non-overlapping cycle loop)
    if negative_cycle_nodes and len(negative_cycle_nodes) >= 2:
        for idx in range(len(negative_cycle_nodes) - 1):
            u_c = negative_cycle_nodes[idx]
            v_c = negative_cycle_nodes[idx + 1]
            if u_c in network.servers and v_c in network.servers:
                s_uc = network.servers[u_c]
                s_vc = network.servers[v_c]
                is_dir = any(l.u == u_c and l.v == v_c and l.directed for l in network.links)
                xs, ys, arr_x, arr_y, arr_ux, arr_uy, _, _ = get_link_geometry(u_c, v_c, is_dir, network)
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color="#ef4444", width=5.5, dash="dash"),
                        hoverinfo="text",
                        hovertext=f"🚨 Negative Cycle Edge: {s_uc.name} → {s_vc.name}",
                        showlegend=False,
                    )
                )
                if is_dir:
                    fig.add_annotation(
                        x=arr_x + 6.0 * arr_ux,
                        y=arr_y + 6.0 * arr_uy,
                        ax=arr_x - 6.0 * arr_ux,
                        ay=arr_y - 6.0 * arr_uy,
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.3,
                        arrowwidth=2.5,
                        arrowcolor="#ef4444",
                    )

    # Edge Latency Badges (Non-overlapping with dark box background)
    if show_edge_labels:
        for link in network.links:
            if link.u in network.servers and link.v in network.servers:
                _, _, _, _, _, _, label_x, label_y = get_link_geometry(link.u, link.v, link.directed, network)
                dir_sym = "→" if link.directed else ""
                badge_bg = "#250a0a" if link.latency < 0 else "#0d1017"
                badge_border = "#dc2626" if link.latency < 0 else "#272e3f"
                badge_text_color = "#fca5a5" if link.latency < 0 else "#94a3b8"

                fig.add_annotation(
                    x=label_x,
                    y=label_y,
                    text=f"<b>{link.latency:.1f}ms{dir_sym}</b>",
                    showarrow=False,
                    font=dict(color=badge_text_color, size=10, family="monospace"),
                    bgcolor=badge_bg,
                    bordercolor=badge_border,
                    borderwidth=1,
                    borderpad=3,
                    opacity=0.95,
                )

    # 2. Draw Server Nodes
    node_x = []
    node_y = []
    node_sizes = []
    node_colors = []
    node_line_colors = []
    node_line_widths = []
    node_labels = []
    hover_texts = []

    for sid, server in network.servers.items():
        node_x.append(server.x)
        node_y.append(server.y)
        node_labels.append(f"<b>{server.name}</b>")

        is_selected = sid in highlight_nodes
        # Size based on capacity (base 24 to 45 px)
        size = 22 + int(math.sqrt(max(10, server.capacity)) * 3.2)
        node_sizes.append(size)

        clients_info = "".join([f"<br>• Client: {c.name} (Demand: {c.demand})" for c in server_clients[sid]])
        if not clients_info:
            clients_info = "<br>• No local clients"

        if is_selected:
            node_colors.append(highlight_node_color)
            node_line_colors.append(THEME["node_selected_border"])
            node_line_widths.append(3.0)
            role_badge = "★ VM HOST SERVER ★"
        else:
            node_colors.append(THEME["node_normal"])
            node_line_colors.append(THEME["node_normal_border"])
            node_line_widths.append(2.0)
            role_badge = "Standard Server"

        hover_texts.append(
            f"<b>{server.name} (ID: {sid})</b><br>"
            f"Role: {role_badge}<br>"
            f"Capacity: <b>{server.capacity} units</b>"
            f"{clients_info}"
        )

    # Server node halos for selected VM host servers
    if highlight_nodes:
        glow_x = [network.servers[sid].x for sid in highlight_nodes if sid in network.servers]
        glow_y = [network.servers[sid].y for sid in highlight_nodes if sid in network.servers]
        glow_sizes = [
            (22 + int(math.sqrt(max(10, network.servers[sid].capacity)) * 3.2)) + 18
            for sid in highlight_nodes if sid in network.servers
        ]
        fig.add_trace(
            go.Scatter(
                x=glow_x,
                y=glow_y,
                mode="markers",
                marker=dict(
                    size=glow_sizes,
                    color=THEME["node_selected_glow"],
                    line=dict(color=THEME["node_selected_border"], width=1.5),
                ),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Base Server Nodes
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_labels,
            textposition="top center",
            textfont=dict(color=THEME["text_bright"], size=11, family="sans-serif"),
            hoverinfo="text",
            hovertext=hover_texts,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(color=node_line_colors, width=node_line_widths),
            ),
            showlegend=False,
        )
    )

    # Layout styling: Deep pitch black theme
    fig.update_layout(
        title=dict(
            text=f"<b style='color: {THEME['text_bright']}'>{title}</b>",
            x=0.04,
            y=0.96,
            font=dict(family="sans-serif", size=16),
        ),
        paper_bgcolor=THEME["bg_pitch_black"],
        plot_bgcolor=THEME["bg_pitch_black"],
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
        ),
        hoverlabel=dict(
            bgcolor="#0b0e14",
            bordercolor=THEME["border_subtle"],
            font=dict(color="#ffffff", size=12, family="monospace"),
        ),
    )

    return fig


def draw_latency_heatmap(
    matrix_df: pd.DataFrame,
    title: str = "All-Pairs Shortest Path Latency Matrix (ms)",
) -> go.Figure:
    """
    Renders an interactive heatmap of all-pairs shortest path latencies
    with black to dark crimson red color gradient.
    """
    # Custom shades of black and dark red colorscale
    colorscale = [
        [0.0, "#050608"],
        [0.2, "#1c0a0a"],
        [0.5, "#450a0a"],
        [0.8, "#7f1d1d"],
        [1.0, "#b91c1c"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_df.values,
            x=matrix_df.columns,
            y=matrix_df.index,
            colorscale=colorscale,
            text=matrix_df.values,
            texttemplate="%{text:.0f}",
            textfont=dict(color="#f8fafc", size=10, family="monospace"),
            hoverongaps=False,
            colorbar=dict(
                title=dict(text="Latency (ms)", font=dict(color="#cbd5e1", size=12)),
                tickfont=dict(color="#94a3b8"),
                outlinecolor="#222938",
                bordercolor="#222938",
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b style='color: {THEME['text_bright']}'>{title}</b>",
            x=0.03,
            y=0.97,
            font=dict(size=15),
        ),
        paper_bgcolor=THEME["bg_pitch_black"],
        plot_bgcolor=THEME["bg_pitch_black"],
        height=520,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(
            tickangle=-40,
            tickfont=dict(color="#cbd5e1", size=11),
            gridcolor="#151922",
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color="#cbd5e1", size=11),
            gridcolor="#151922",
        ),
    )
    return fig


def draw_comparison_charts(
    comparison_df: pd.DataFrame,
) -> Tuple[go.Figure, go.Figure]:
    """
    Renders comparative bar charts for Total Delay and Runtime/Host Count
    styled in shades of black and dark red.
    """
    algos = comparison_df["Algorithm"].tolist()

    if "Total Value (Points)" in comparison_df.columns:
        vals = [int(v) if str(v).isdigit() else int(float(str(v))) for v in comparison_df["Total Value (Points)"]]
        
        # Figure 1: Total Value Benchmark
        fig_val = go.Figure()
        fig_val.add_trace(
            go.Bar(
                name="Total Value (Points)",
                x=algos,
                y=vals,
                marker=dict(color=["#dc2626", "#991b1b", "#7f1d1d", "#3f3f46"][:len(algos)]),
                text=[f"{v} pts" for v in vals],
                textposition="auto",
            )
        )
        fig_val.update_layout(
            title=dict(
                text="<b style='color: #f8fafc'>Total Task Value (Points) — Higher is Better (DP Optimum)</b>",
                x=0.03,
                font=dict(size=14),
            ),
            paper_bgcolor=THEME["bg_pitch_black"],
            plot_bgcolor=THEME["bg_pitch_black"],
            height=360,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(tickfont=dict(color="#f8fafc", size=11), gridcolor="#151922"),
            yaxis=dict(
                title=dict(text="Total Value (Points)", font=dict(color="#94a3b8")),
                tickfont=dict(color="#94a3b8"),
                gridcolor="#151922",
            ),
        )

        # Figure 2: Capacity Utilization (%)
        if "Capacity Used (%)" in comparison_df.columns:
            util_vals = [
                float(str(v).replace("%", "").strip()) if str(v) != "N/A" else 0.0
                for v in comparison_df["Capacity Used (%)"]
            ]
            fig_util = go.Figure()
            fig_util.add_trace(
                go.Bar(
                    name="Capacity Used (%)",
                    x=algos,
                    y=util_vals,
                    marker=dict(color=["#dc2626", "#991b1b", "#7f1d1d", "#3f3f46"][:len(algos)]),
                    text=[f"{v:.1f}%" for v in util_vals],
                    textposition="auto",
                )
            )
            fig_util.update_layout(
                title=dict(
                    text="<b style='color: #f8fafc'>Server Capacity Utilization (%)</b>",
                    x=0.03,
                    font=dict(size=14),
                ),
                paper_bgcolor=THEME["bg_pitch_black"],
                plot_bgcolor=THEME["bg_pitch_black"],
                height=360,
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis=dict(tickfont=dict(color="#f8fafc", size=11), gridcolor="#151922"),
                yaxis=dict(
                    title=dict(text="Utilization (%)", font=dict(color="#94a3b8")),
                    tickfont=dict(color="#94a3b8"),
                    range=[0, 105],
                    gridcolor="#151922",
                ),
            )
        else:
            fig_util = go.Figure()

        return fig_val, fig_util

    if "Total Delay (ms)" in comparison_df.columns:
        delays = [float(str(v).replace("ms", "").strip()) if str(v) != "N/A" else 0.0 for v in comparison_df["Total Delay (ms)"]]
        
        # Figure 1: Total Delay Benchmark
        fig_lat = go.Figure()
        fig_lat.add_trace(
            go.Bar(
                name="Total Delay (ms)",
                x=algos,
                y=delays,
                marker=dict(color=["#8b0000", "#dc2626", "#52525b"]),
                text=[f"{v:.1f} ms" if v > 0 else "N/A" for v in delays],
                textposition="auto",
            )
        )
        fig_lat.update_layout(
            title=dict(
                text="<b style='color: #f8fafc'>Total Delay Benchmark (Lower is Better)</b>",
                x=0.03,
                font=dict(size=14),
            ),
            paper_bgcolor=THEME["bg_pitch_black"],
            plot_bgcolor=THEME["bg_pitch_black"],
            height=360,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(tickfont=dict(color="#f8fafc", size=11), gridcolor="#151922"),
            yaxis=dict(
                title=dict(text="Total Delay (ms)", font=dict(color="#94a3b8")),
                tickfont=dict(color="#94a3b8"),
                gridcolor="#151922",
            ),
        )

        # Figure 2: Server Host Count or Runtime
        if "Runtime (ms)" in comparison_df.columns:
            runtimes = [float(v) if str(v) != "N/A" else 0.0 for v in comparison_df["Runtime (ms)"]]
            fig_util = go.Figure()
            fig_util.add_trace(
                go.Bar(
                    name="Runtime (ms)",
                    x=algos,
                    y=runtimes,
                    marker=dict(color=["#8b0000", "#991b1b", "#52525b"]),
                    text=[f"{v:.2f} ms" for v in runtimes],
                    textposition="auto",
                )
            )
            fig_util.update_layout(
                title=dict(
                    text="<b style='color: #f8fafc'>Execution Time (ms)</b>",
                    x=0.03,
                    font=dict(size=14),
                ),
                paper_bgcolor=THEME["bg_pitch_black"],
                plot_bgcolor=THEME["bg_pitch_black"],
                height=360,
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis=dict(tickfont=dict(color="#f8fafc", size=11), gridcolor="#151922"),
                yaxis=dict(
                    title=dict(text="Time (ms)", font=dict(color="#94a3b8")),
                    tickfont=dict(color="#94a3b8"),
                    gridcolor="#151922",
                ),
            )
        else:
            fig_util = go.Figure()
        return fig_lat, fig_util

    avg_lat = [float(v) if v != "N/A" else 0.0 for v in comparison_df.get("Avg Latency (ms)", [])]
    max_lat = [float(v) if v != "N/A" else 0.0 for v in comparison_df.get("Max Latency (ms)", [])]
    util_vals = [
        float(str(v).replace("%", "")) if v != "N/A" else 0.0
        for v in comparison_df.get("Avg Utilization (%)", [])
    ]

    # Figure 1: Latency Benchmark
    fig_lat = go.Figure()
    fig_lat.add_trace(
        go.Bar(
            name="Weighted Avg Latency (ms)",
            x=algos,
            y=avg_lat,
            marker=dict(color=["#8b0000", "#7f1d1d", "#3f3f46"]),
            text=[f"{v:.1f}ms" if v > 0 else "N/A" for v in avg_lat],
            textposition="auto",
        )
    )
    fig_lat.add_trace(
        go.Bar(
            name="Max Client Latency (ms)",
            x=algos,
            y=max_lat,
            marker=dict(color=["#dc2626", "#b91c1c", "#52525b"]),
            text=[f"{v:.1f}ms" if v > 0 else "N/A" for v in max_lat],
            textposition="auto",
        )
    )

    fig_lat.update_layout(
        title=dict(
            text="<b style='color: #f8fafc'>Latency Benchmark: DP vs Greedy vs Random (Lower is Better)</b>",
            x=0.03,
            font=dict(size=14),
        ),
        paper_bgcolor=THEME["bg_pitch_black"],
        plot_bgcolor=THEME["bg_pitch_black"],
        barmode="group",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            font=dict(color="#cbd5e1"),
            bgcolor="#0d0f14",
            bordercolor="#222938",
            borderwidth=1,
        ),
        xaxis=dict(tickfont=dict(color="#f8fafc", size=11), gridcolor="#151922"),
        yaxis=dict(
            title=dict(text="Latency (ms)", font=dict(color="#94a3b8")),
            tickfont=dict(color="#94a3b8"),
            gridcolor="#151922",
        ),
    )

    # Figure 2: Capacity Utilization
    fig_util = go.Figure()
    fig_util.add_trace(
        go.Bar(
            x=algos,
            y=util_vals,
            marker=dict(color=["#8b0000", "#991b1b", "#52525b"]),
            text=[f"{v:.1f}%" if v > 0 else "N/A" for v in util_vals],
            textposition="auto",
        )
    )

    fig_util.update_layout(
        title=dict(
            text="<b style='color: #f8fafc'>Average Capacity Utilization (%)</b>",
            x=0.03,
            font=dict(size=14),
        ),
        paper_bgcolor=THEME["bg_pitch_black"],
        plot_bgcolor=THEME["bg_pitch_black"],
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(tickfont=dict(color="#f8fafc", size=11), gridcolor="#151922"),
        yaxis=dict(
            title=dict(text="Utilization (%)", font=dict(color="#94a3b8")),
            tickfont=dict(color="#94a3b8"),
            range=[0, 105],
            gridcolor="#151922",
        ),
    )

    return fig_lat, fig_util
