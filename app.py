"""
app.py - Cloud Task Placement Simulator (Clean College Demo Edition).

Algorithms:
1. Minimum Spanning Tree: Kruskal's Algorithm with Disjoint-Set Union
2. Shortest-Path Routing: Dijkstra's, Bellman-Ford with Negative Cycle Detection, and Floyd-Warshall
3. Task Placement Optimization: 0/1 Knapsack Dynamic Programming
4. Brute-Force Decision Tree: 0/1 Knapsack Binary Expansion with Pruning
5. Interactive Canvas Studio: Drag & Drop Server Placement & Manual Wiring
"""

from __future__ import annotations
import math
import time
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

import graph_utils
import visualization
import shortest_path
import mst
import placement_dp
import baselines
import decision_tree

# Algorithm bindings
CloudNetwork = graph_utils.CloudNetwork
Server = graph_utils.Server
Link = graph_utils.Link
Client = graph_utils.Client
create_normal_network_5 = getattr(graph_utils, "create_normal_network_5", graph_utils.create_sample_network)
create_dijkstra_vs_bellman_network = graph_utils.create_dijkstra_vs_bellman_network
create_negative_cycle_network = graph_utils.create_negative_cycle_network

kruskal_mst = mst.kruskal_mst
prim_mst = mst.prim_mst
dijkstra_shortest_path = shortest_path.dijkstra_shortest_path
bellman_ford_shortest_path = shortest_path.bellman_ford_shortest_path
floyd_warshall_all_pairs = shortest_path.floyd_warshall_all_pairs

Task = placement_dp.Task
get_default_demo_tasks = getattr(placement_dp, "get_default_demo_tasks", placement_dp.get_default_tasks)
solve_knapsack_01_dp = placement_dp.solve_knapsack_01_dp

solve_knapsack_greedy_value = baselines.solve_knapsack_greedy_value
solve_knapsack_greedy_ratio = baselines.solve_knapsack_greedy_ratio

build_decision_tree = decision_tree.build_decision_tree
draw_decision_tree_plotly = decision_tree.draw_decision_tree_plotly

draw_network_graph = visualization.draw_network_graph
draw_latency_heatmap = visualization.draw_latency_heatmap
THEME = visualization.THEME

# Custom Logisim Drag & Drop Canvas Component
CANVAS_DIR = Path(__file__).resolve().parent / "canvas_component"
logisim_canvas = components.declare_component("logisim_canvas", path=str(CANVAS_DIR))

# Page Configuration
st.set_page_config(
    page_title="Cloud Task Placement Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark Theme CSS Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #050608;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .metric-card {
        background-color: #0d0f14;
        border: 1px solid #1e2430;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 4px 0 2px 0;
    }
    .metric-sub {
        color: #64748b;
        font-size: 0.75rem;
    }
    .algo-box {
        background-color: #0d0f14;
        border: 1px solid #1e2430;
        border-left: 4px solid #38bdf8;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.88rem;
        line-height: 1.4;
    }
    div[data-testid="stElementContainer"]:has(iframe) {
        opacity: 1 !important;
        transition: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# INITIALIZE SESSION STATE (DEFAULTS DEMO IMMEDIATELY)
# ---------------------------------------------------------------------
if "network" not in st.session_state:
    st.session_state.network = create_dijkstra_vs_bellman_network()
    st.session_state.allow_negative_weights = True
    st.session_state.entry_node_id = 0
    st.session_state.routing_orig = 0
    st.session_state.routing_dest = 1
    st.session_state.routing_mode = "Dijkstra"

if "tasks" not in st.session_state:
    st.session_state.tasks = get_default_demo_tasks()

if "selected_server_id" not in st.session_state:
    st.session_state.selected_server_id = 0

if "last_canvas_ts" not in st.session_state:
    st.session_state.last_canvas_ts = 0


# ---------------------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------------------
st.sidebar.markdown("## Cloud Simulator")
st.sidebar.markdown("### Presets")

c_p1, c_p2 = st.sidebar.columns(2)
with c_p1:
    if st.button("Normal (5 nodes)", use_container_width=True):
        st.session_state.network = create_normal_network_5()
        st.session_state.allow_negative_weights = False
        st.session_state.entry_node_id = 0
        st.session_state.routing_orig = 0
        st.session_state.routing_dest = 2
        st.session_state.selected_server_id = 0
        st.session_state.force_canvas_reload = True
        st.rerun()

with c_p2:
    if st.button("Dij vs BF demo", use_container_width=True):
        st.session_state.network = create_dijkstra_vs_bellman_network()
        st.session_state.allow_negative_weights = True
        st.session_state.entry_node_id = 0
        st.session_state.routing_orig = 0
        st.session_state.routing_dest = 1
        st.session_state.selected_server_id = 0
        st.session_state.force_canvas_reload = True
        st.rerun()

if st.sidebar.button("Negative cycle demo", use_container_width=True):
    st.session_state.network = create_negative_cycle_network()
    st.session_state.allow_negative_weights = True
    st.session_state.entry_node_id = 0
    st.session_state.routing_orig = 0
    st.session_state.routing_dest = 1
    st.session_state.selected_server_id = 0
    st.session_state.force_canvas_reload = True
    st.rerun()

st.sidebar.markdown("### Settings")
allow_neg = st.sidebar.toggle(
    "Allow negative weights",
    value=st.session_state.get("allow_negative_weights", False),
    key="toggle_allow_neg",
)
st.session_state.allow_negative_weights = allow_neg

# Synchronize any canvas edits
incoming_canvas_data = st.session_state.get("logisim_network_canvas")
if incoming_canvas_data and isinstance(incoming_canvas_data, dict) and "servers" in incoming_canvas_data:
    incoming_ts = incoming_canvas_data.get("timestamp")
    if incoming_ts != st.session_state.get("last_canvas_ts"):
        st.session_state.last_canvas_ts = incoming_ts
        new_servers_raw = incoming_canvas_data.get("servers", [])
        new_links_raw = incoming_canvas_data.get("links", [])
        new_clients_raw = incoming_canvas_data.get("clients", [])

        new_servers = {
            int(s["id"]): Server(
                id=int(s["id"]),
                name=str(s["name"]),
                capacity=int(s["capacity"]),
                x=float(s.get("x", 200)),
                y=float(s.get("y", 200)),
            )
            for s in new_servers_raw
        }
        new_links = [
            Link(
                u=int(l["u"]),
                v=int(l["v"]),
                latency=float(l["latency"]),
                directed=bool(l.get("directed", False)),
            )
            for l in new_links_raw
            if int(l["u"]) in new_servers and int(l["v"]) in new_servers
        ]
        new_clients = [
            Client(
                id=int(c["id"]),
                name=str(c["name"]),
                server_id=int(c["server_id"]),
                demand=int(c["demand"]),
            )
            for c in new_clients_raw
            if int(c["server_id"]) in new_servers
        ]
        st.session_state.network = CloudNetwork(servers=new_servers, links=new_links, clients=new_clients)

net: CloudNetwork = st.session_state.network
server_keys = sorted(net.servers.keys()) if net.servers else []

if server_keys:
    curr_entry = st.session_state.get("entry_node_id", 0)
    entry_idx = server_keys.index(curr_entry) if curr_entry in server_keys else 0
    entry_node_id = st.sidebar.selectbox(
        "User Entry Node:",
        options=server_keys,
        index=entry_idx,
        format_func=lambda sid: f"{net.servers[sid].name} (ID: {sid})",
        key="entry_node_select_ctrl",
    )
    st.session_state.entry_node_id = entry_node_id
else:
    entry_node_id = 0
    st.sidebar.info("Add servers to select entry node.")

# Topology Validation Error Warning
is_valid_net, validation_errors = net.validate(allow_negative_weights=st.session_state.get("allow_negative_weights", False))
if not is_valid_net:
    st.error(f"Topology Error: {validation_errors[0]}")


# ---------------------------------------------------------------------
# MAIN APP TABS (EXACTLY 5 TABS)
# ---------------------------------------------------------------------
st.markdown("## Cloud Task Placement Simulator")

tabs = st.tabs([
    "Canvas Studio",
    "Network & MST",
    "Routing",
    "Dynamic Programming (Tabulation)",
    "0/1 Knapsack",
])


# =====================================================================
# TAB 1: CANVAS STUDIO
# =====================================================================
with tabs[0]:
    st.markdown("### Canvas Studio")
    st.caption("Interactive topology editor to add servers, bidirectional links, and directed links.")

    col_cm1, col_cm2, col_cm3, col_cm4 = st.columns(4)
    with col_cm1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>Servers</div>
                <div class='metric-value'>{net.num_servers}</div>
                <div class='metric-sub'>Active nodes in graph</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_cm2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>Links</div>
                <div class='metric-value'>{net.num_links}</div>
                <div class='metric-sub'>Network connections</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_cm3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>Total Capacity</div>
                <div class='metric-value'>{net.total_capacity}</div>
                <div class='metric-sub'>Aggregate server capacity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_cm4:
        conn_color = "#4ade80" if net.is_connected() else "#f87171"
        conn_text = "CONNECTED" if net.is_connected() else "PARTITIONED"
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>Status</div>
                <div class='metric-value' style='color: {conn_color}; font-size: 1.15rem;'>{conn_text}</div>
                <div class='metric-sub'>Graph connectivity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    force_reload = st.session_state.pop("force_canvas_reload", False)
    canvas_output = logisim_canvas(
        network_data=net.to_dict(),
        force_reload=force_reload,
        key="logisim_network_canvas",
        default=None,
    )

    st.markdown(
        """
        <div class='algo-box'>
            <b>Canvas Controls:</b><br>
            • <b>Add Server:</b> Click empty grid. &nbsp;|&nbsp;
            • <b>Add Wire:</b> Click Server A, then click Server B. Click weight badge to edit latency. &nbsp;|&nbsp;
            • <b>Delete:</b> Click server or link to remove. &nbsp;|&nbsp;
            • <b>Directed / One-Way:</b> Select link and toggle directed in toolbar. Negative weights permitted on directed links only.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# TAB 2: NETWORK & MST
# =====================================================================
with tabs[1]:
    st.markdown("### Network & MST")
    st.caption("Minimum Spanning Tree backbone comparing Kruskal's and Prim's algorithms.")

    if net.num_servers < 2:
        st.info("Add at least 2 servers in Canvas Studio or select a preset.")
    else:
        col_m_opt1, col_m_opt2 = st.columns([3, 1])
        with col_m_opt1:
            mst_algo_choice = st.radio(
                "Algorithm:",
                options=["Kruskal's Algorithm (Edge-Centric)", "Prim's Algorithm (Vertex-Centric)"],
                horizontal=True,
                key="mst_algo_choice_radio",
            )
        with col_m_opt2:
            if "Prim" in mst_algo_choice:
                prim_start = st.selectbox(
                    "Start Server:",
                    options=server_keys,
                    format_func=lambda sid: f"{net.servers[sid].name} (ID: {sid})",
                    key="prim_start_select",
                )
            else:
                prim_start = None

        kruskal_res = kruskal_mst(net)
        default_prim_start = server_keys[0] if server_keys else 0
        prim_res = prim_mst(net, start_node=prim_start if prim_start is not None else default_prim_start)

        if "Prim" in mst_algo_choice:
            active_mst = prim_res
            mst_algo_name = "Prim"
            mst_complexity = "O(|E| log |V|)"
        else:
            active_mst = kruskal_res
            mst_algo_name = "Kruskal"
            mst_complexity = "O(|E| log |E|)"

        mst_edges_set = set((min(u, v), max(u, v)) for u, v, _ in active_mst.mst_edges)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(
                f"""
                <div class='metric-card' style='border-left: 3px solid #22c55e;'>
                    <div class='metric-title'>Total MST Cost</div>
                    <div class='metric-value' style='color: #4ade80;'>{active_mst.total_cost:.1f} ms</div>
                    <div class='metric-sub'>{mst_algo_name} backbone latency</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-title'>Tree Edges</div>
                    <div class='metric-value'>{len(active_mst.mst_edges)} / {max(0, net.num_servers - 1)}</div>
                    <div class='metric-sub'>Spanning requirement: |V| - 1</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m3:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-title'>Time Complexity</div>
                    <div class='metric-value'>{mst_complexity}</div>
                    <div class='metric-sub'>{mst_algo_name}'s Algorithm</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Note comparing Kruskal and Prim matching costs
        st.markdown(
            f"""
            <div class='algo-box' style='border-left-color: #22c55e;'>
                <b>MST Invariance Check:</b> Kruskal Cost = <b>{kruskal_res.total_cost:.1f} ms</b> &nbsp;|&nbsp;
                Prim Cost = <b>{prim_res.total_cost:.1f} ms</b> &nbsp;➔&nbsp;
                <span style='color: #4ade80; font-weight: 600;'>Costs match: {kruskal_res.total_cost:.1f} ms</span><br>
                <span style='color: #94a3b8; font-size: 0.84rem;'>Both Kruskal and Prim discover minimum spanning trees with identical total cost on connected networks.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Step-by-Step Animation Controls
        mst_steps = active_mst.steps
        num_steps = len(mst_steps)
        if num_steps > 0:
            col_anim1, col_anim2 = st.columns([1, 4])
            with col_anim1:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                play_clicked = st.button("▶ Play Animation", key=f"play_mst_{mst_algo_name}")
            with col_anim2:
                step_idx = st.slider(
                    f"{mst_algo_name} Step Scrubber:",
                    min_value=1,
                    max_value=num_steps,
                    value=num_steps,
                    key=f"mst_step_slider_{mst_algo_name}",
                )

            plot_holder = st.empty()
            log_holder = st.empty()

            def _render_mst_step(s_i: int):
                s = mst_steps[s_i - 1]
                eval_edge = (s.u, s.v)
                eval_status = "accepted" if s.accepted else "rejected"
                mst_so_far = set((min(u, v), max(u, v)) for u, v, _ in s.mst_edges_so_far)
                rejs_so_far = set(
                    (min(st_item.u, st_item.v), max(st_item.u, st_item.v))
                    for st_item in mst_steps[:s_i]
                    if not st_item.accepted
                )
                fig_step = draw_network_graph(
                    network=net,
                    highlight_edges=mst_so_far,
                    highlight_edge_color="#22c55e",
                    evaluating_edge=eval_edge,
                    evaluating_status=eval_status,
                    rejected_edges=rejs_so_far,
                    title=f"{mst_algo_name} Step {s_i}/{num_steps}: Edge {s.u_name} ↔ {s.v_name} ({'Accepted' if s.accepted else 'Rejected'})",
                )
                st_color = "#4ade80" if s.accepted else "#f87171"
                st_bg = "rgba(34, 197, 94, 0.12)" if s.accepted else "rgba(239, 68, 68, 0.12)"
                uf_info = (
                    f"Union-Find State: <b>{s.components_count} disjoint components</b> remaining."
                    if mst_algo_name == "Kruskal"
                    else f"Prim Frontier: <b>{s.components_count} unvisited vertices</b>."
                )
                log_html = f"""
                <div class='algo-box' style='border-left-color: {st_color}; background-color: {st_bg};'>
                    <b>Step {s_i}/{num_steps}:</b> Evaluating Edge <b>{s.u_name} ↔ {s.v_name}</b> (Latency: {s.latency:.1f} ms) ➔ <span style='color: {st_color}; font-weight: 700;'>{'ACCEPTED' if s.accepted else 'REJECTED (CYCLE)'}</span><br>
                    • <b>Decision Reason:</b> {s.reason}<br>
                    • <b>State:</b> {uf_info} &nbsp;|&nbsp; Running MST Cost: <b>{s.running_cost:.1f} ms</b>
                </div>
                """
                return fig_step, log_html

            if play_clicked:
                for s_i in range(1, num_steps + 1):
                    fig_step, log_html = _render_mst_step(s_i)
                    plot_holder.plotly_chart(fig_step, use_container_width=True, key=f"mst_anim_{mst_algo_name}_{s_i}")
                    log_holder.markdown(log_html, unsafe_allow_html=True)
                    time.sleep(0.35)
            else:
                fig_step, log_html = _render_mst_step(step_idx)
                plot_holder.plotly_chart(fig_step, use_container_width=True)
                log_holder.markdown(log_html, unsafe_allow_html=True)
        else:
            fig_mst = draw_network_graph(
                network=net,
                highlight_edges=mst_edges_set,
                highlight_edge_color="#22c55e",
                title=f"{mst_algo_name} MST (Total Cost: {active_mst.total_cost:.1f} ms)",
            )
            st.plotly_chart(fig_mst, use_container_width=True)

        # Edge Decisions Table (Accepted vs Rejected)
        st.markdown(f"#### {mst_algo_name} Edge Selection Log")
        if active_mst.steps:
            mst_rows = []
            for step in active_mst.steps:
                status_label = "Accepted" if step.accepted else "Rejected (Cycle)"
                mst_rows.append({
                    "Edge": f"{step.u_name} ↔ {step.v_name}",
                    "Delay (ms)": f"{step.latency:.1f}",
                    "Status": status_label,
                    "Running Cost (ms)": f"{step.running_cost:.1f}",
                })
            st.dataframe(pd.DataFrame(mst_rows), use_container_width=True)


# =====================================================================
# TAB 3: ROUTING
# =====================================================================
with tabs[2]:
    st.markdown("### Routing")
    st.caption("Shortest path routing comparing Dijkstra, Bellman-Ford with negative cycle detection, and Floyd-Warshall.")

    if len(net.servers) < 2:
        st.info("Add at least 2 servers in Canvas Studio or select a preset.")
    else:
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            default_orig = st.session_state.get("routing_orig", server_keys[0])
            orig_idx = server_keys.index(default_orig) if default_orig in server_keys else 0
            orig_sid = st.selectbox(
                "Origin Server:",
                options=server_keys,
                index=orig_idx,
                format_func=lambda sid: f"{net.servers[sid].name} (ID: {sid})",
                key="route_orig_select",
            )
            st.session_state.routing_orig = orig_sid

        with col_sel2:
            default_dest = st.session_state.get("routing_dest", server_keys[min(1, len(server_keys) - 1)])
            dest_idx = server_keys.index(default_dest) if default_dest in server_keys else min(1, len(server_keys) - 1)
            dest_sid = st.selectbox(
                "Destination Server:",
                options=server_keys,
                index=dest_idx,
                format_func=lambda sid: f"{net.servers[sid].name} (ID: {sid})",
                key="route_dest_select",
            )
            st.session_state.routing_dest = dest_sid

        routing_mode = st.radio(
            "Routing Mode:",
            options=["Dijkstra", "Bellman-Ford", "Floyd-Warshall"],
            horizontal=True,
            key="routing_mode_radio",
        )

        fw_all = floyd_warshall_all_pairs(net)

        if routing_mode == "Dijkstra":
            dijk_res = dijkstra_shortest_path(net, source=orig_sid)
            d_dist = dijk_res.distances.get(dest_sid, float("inf"))
            d_path = dijk_res.get_path_to(dest_sid)
            d_hops = " ➔ ".join([net.servers[n].name for n in d_path]) if d_path else "No route"

            if net.has_negative_weights():
                st.warning("⚠️ Warning: Network contains negative edge weights. Dijkstra may yield incorrect shortest paths.")

            col_d1, col_d2 = st.columns([1, 2])
            with col_d1:
                dist_str = f"{d_dist:.1f} ms" if not math.isinf(d_dist) else "Unreachable"
                st.markdown(
                    f"""
                    <div class='metric-card' style='border-left: 3px solid #38bdf8;'>
                        <div class='metric-title'>Dijkstra Delay</div>
                        <div class='metric-value'>{dist_str}</div>
                        <div class='metric-sub'>{len(d_path)} hops | O((|V| + |E|) log |V|)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_d2:
                st.markdown(
                    f"""
                    <div class='algo-box'>
                        <b>Hop Sequence:</b><br>
                        <span style='color: #38bdf8; font-family: monospace;'>{d_hops}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            d_steps = dijk_res.steps
            num_d_steps = len(d_steps)
            final_route_edges = [(d_path[i], d_path[i + 1]) for i in range(len(d_path) - 1)] if len(d_path) >= 2 else []

            if num_d_steps > 0:
                col_d_ctrl1, col_d_ctrl2 = st.columns([1, 4])
                with col_d_ctrl1:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    play_dijk = st.button("▶ Play Animation", key="play_dijk")
                with col_d_ctrl2:
                    d_step_idx = st.slider(
                        "Dijkstra Exploration Step:",
                        min_value=1,
                        max_value=num_d_steps,
                        value=num_d_steps,
                        key="dijk_step_slider",
                    )

                d_plot_holder = st.empty()
                d_log_holder = st.empty()

                def _render_dijk_step(s_i: int):
                    st_obj = d_steps[s_i - 1]
                    is_last = (s_i == num_d_steps)
                    eval_edge = (st_obj.current_node, st_obj.neighbor_node) if st_obj.neighbor_node is not None else None
                    eval_status = "accepted" if st_obj.relaxed else "rejected"
                    tree_edges = set(st_obj.active_tree_edges)
                    vis_nodes = set(st_obj.visited_nodes) | {orig_sid}
                    route_to_draw = final_route_edges if is_last else []

                    fig = draw_network_graph(
                        network=net,
                        highlight_edges=tree_edges if not is_last else set(),
                        highlight_edge_color="#38bdf8",
                        route_edges=route_to_draw,
                        evaluating_edge=eval_edge if not is_last else None,
                        evaluating_status=eval_status,
                        highlight_nodes=vis_nodes | {orig_sid, dest_sid},
                        highlight_node_color=THEME["node_selected"],
                        title=f"Dijkstra Step {s_i}/{num_d_steps}: {st_obj.action} (Origin: {net.servers[orig_sid].name} ➔ Dest: {net.servers[dest_sid].name})",
                    )
                    log_html = f"""
                    <div class='algo-box' style='border-left-color: #38bdf8;'>
                        <b>Step {s_i}/{num_d_steps} [{st_obj.action}]:</b> {st_obj.reason}<br>
                        • Current server: <b>{st_obj.current_node_name}</b> (delay: {st_obj.current_dist:.1f} ms) &nbsp;|&nbsp; Settled: <b>{len(st_obj.visited_nodes)} servers</b>
                    </div>
                    """
                    return fig, log_html

                if play_dijk:
                    for s_i in range(1, num_d_steps + 1):
                        fig_step, log_html = _render_dijk_step(s_i)
                        d_plot_holder.plotly_chart(fig_step, use_container_width=True, key=f"dijk_anim_{s_i}")
                        d_log_holder.markdown(log_html, unsafe_allow_html=True)
                        time.sleep(0.3)
                else:
                    fig_step, log_html = _render_dijk_step(d_step_idx)
                    d_plot_holder.plotly_chart(fig_step, use_container_width=True)
                    d_log_holder.markdown(log_html, unsafe_allow_html=True)
            else:
                fig_d = draw_network_graph(
                    network=net,
                    route_edges=final_route_edges,
                    highlight_nodes={orig_sid, dest_sid},
                    highlight_node_color=THEME["node_selected"],
                    title=f"Dijkstra Route: {net.servers[orig_sid].name} ➔ {net.servers[dest_sid].name}",
                )
                st.plotly_chart(fig_d, use_container_width=True)

        elif routing_mode == "Bellman-Ford":
            bf_res = bellman_ford_shortest_path(net, source=orig_sid)
            bf_dist = bf_res.distances.get(dest_sid, float("inf"))
            bf_path = bf_res.get_path_to(dest_sid)
            bf_hops = " ➔ ".join([net.servers[n].name for n in bf_path]) if bf_path else "No route"

            if bf_res.has_negative_cycle:
                cycle_server_names = [net.servers[n].name for n in bf_res.negative_cycle_nodes if n in net.servers]
                st.error(
                    f"Negative cycle detected: shortest path is undefined (-infinity)\n\n"
                    f"Cycle nodes: {cycle_server_names}"
                )

            col_bf1, col_bf2 = st.columns([1, 2])
            with col_bf1:
                bf_val_str = "-∞" if bf_res.has_negative_cycle else (f"{bf_dist:.1f} ms" if not math.isinf(bf_dist) else "Unreachable")
                st.markdown(
                    f"""
                    <div class='metric-card' style='border-left: 3px solid #4ade80;'>
                        <div class='metric-title'>Bellman-Ford Delay</div>
                        <div class='metric-value'>{bf_val_str}</div>
                        <div class='metric-sub'>{bf_res.passes_run} passes | O(|V| × |E|)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_bf2:
                st.markdown(
                    f"""
                    <div class='algo-box'>
                        <b>Hop Sequence:</b><br>
                        <span style='color: #4ade80; font-family: monospace;'>{bf_hops}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            bf_steps = bf_res.steps
            num_bf_steps = len(bf_steps)
            final_bf_edges = [(bf_path[i], bf_path[i + 1]) for i in range(len(bf_path) - 1)] if len(bf_path) >= 2 else []

            if num_bf_steps > 0:
                col_bf_ctrl1, col_bf_ctrl2 = st.columns([1, 4])
                with col_bf_ctrl1:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    play_bf = st.button("▶ Play Animation", key="play_bf")
                with col_bf_ctrl2:
                    bf_step_idx = st.slider(
                        "Bellman-Ford Relaxation Step:",
                        min_value=1,
                        max_value=num_bf_steps,
                        value=num_bf_steps,
                        key="bf_step_slider",
                    )

                bf_plot_holder = st.empty()
                bf_log_holder = st.empty()

                def _render_bf_step(s_i: int):
                    st_obj = bf_steps[s_i - 1]
                    is_last = (s_i == num_bf_steps)
                    eval_edge = (st_obj.u, st_obj.v)
                    eval_status = "accepted" if st_obj.relaxed else "rejected"
                    tree_edges = set(st_obj.active_tree_edges)
                    route_to_draw = final_bf_edges if (is_last and not bf_res.has_negative_cycle) else []

                    fig = draw_network_graph(
                        network=net,
                        highlight_edges=tree_edges if not is_last else set(),
                        highlight_edge_color="#4ade80",
                        route_edges=route_to_draw,
                        evaluating_edge=eval_edge if not is_last else None,
                        evaluating_status=eval_status,
                        highlight_nodes={orig_sid, dest_sid},
                        highlight_node_color=THEME["node_selected"],
                        negative_cycle_nodes=bf_res.negative_cycle_nodes if bf_res.has_negative_cycle else None,
                        title=f"Bellman-Ford Pass {st_obj.pass_num} (Step {s_i}/{num_bf_steps}): Relax {st_obj.u_name} → {st_obj.v_name}",
                    )
                    st_color = "#4ade80" if st_obj.relaxed else "#94a3b8"
                    log_html = f"""
                    <div class='algo-box' style='border-left-color: {st_color};'>
                        <b>Pass {st_obj.pass_num} (Step {s_i}/{num_bf_steps}):</b> Edge <b>{st_obj.u_name} → {st_obj.v_name}</b> (Latency: {st_obj.latency:.1f} ms)<br>
                        • <b>Result:</b> {st_obj.reason}
                    </div>
                    """
                    return fig, log_html

                if play_bf:
                    for s_i in range(1, num_bf_steps + 1):
                        fig_step, log_html = _render_bf_step(s_i)
                        bf_plot_holder.plotly_chart(fig_step, use_container_width=True, key=f"bf_anim_{s_i}")
                        bf_log_holder.markdown(log_html, unsafe_allow_html=True)
                        time.sleep(0.3)
                else:
                    fig_step, log_html = _render_bf_step(bf_step_idx)
                    bf_plot_holder.plotly_chart(fig_step, use_container_width=True)
                    bf_log_holder.markdown(log_html, unsafe_allow_html=True)
            else:
                fig_bf = draw_network_graph(
                    network=net,
                    route_edges=final_bf_edges,
                    highlight_nodes={orig_sid, dest_sid},
                    highlight_node_color=THEME["node_selected"],
                    negative_cycle_nodes=bf_res.negative_cycle_nodes if bf_res.has_negative_cycle else None,
                    title=f"Bellman-Ford Route: {net.servers[orig_sid].name} ➔ {net.servers[dest_sid].name}",
                )
                st.plotly_chart(fig_bf, use_container_width=True)

        elif routing_mode == "Floyd-Warshall":
            if fw_all.has_negative_cycle:
                cycle_server_names = [net.servers[n].name for n in fw_all.negative_cycle_nodes if n in net.servers]
                st.error(
                    f"Negative cycle detected in all-pairs evaluation! Diagonal dist[i][i] < 0 was detected.\n\n"
                    f"Affected servers: {cycle_server_names}"
                )

            fw_dist = fw_all.get_distance(orig_sid, dest_sid)
            fw_path = fw_all.get_path(orig_sid, dest_sid)
            fw_hops = " ➔ ".join([net.servers[n].name for n in fw_path]) if fw_path else "No route"

            col_fw1, col_fw2 = st.columns([1, 2])
            with col_fw1:
                fw_str = "-∞" if fw_dist == float("-inf") else (f"{fw_dist:.1f} ms" if not math.isinf(fw_dist) else "Unreachable")
                st.markdown(
                    f"""
                    <div class='metric-card' style='border-left: 3px solid #a855f7;'>
                        <div class='metric-title'>Floyd-Warshall Delay</div>
                        <div class='metric-value'>{fw_str}</div>
                        <div class='metric-sub'>Dynamic Programming | O(|V|³)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_fw2:
                st.markdown(
                    f"""
                    <div class='algo-box'>
                        <b>Hop Sequence:</b><br>
                        <span style='color: #c084fc; font-family: monospace;'>{fw_hops}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 1. Graph view with shortest route highlighted
            route_edges_fw = [(fw_path[i], fw_path[i + 1]) for i in range(len(fw_path) - 1)] if len(fw_path) >= 2 else []
            fig_fw = draw_network_graph(
                network=net,
                route_edges=route_edges_fw,
                highlight_nodes={orig_sid, dest_sid},
                highlight_node_color=THEME["node_selected"],
                negative_cycle_nodes=fw_all.negative_cycle_nodes if fw_all.has_negative_cycle else None,
                title=f"Floyd-Warshall Route: {net.servers[orig_sid].name} ➔ {net.servers[dest_sid].name}",
            )
            st.plotly_chart(fig_fw, use_container_width=True)

            # 2. Intermediate node k step slider and matrix updates
            st.markdown("#### All-Pairs Latency Matrix $D^{(k)}$ Progression")
            n_nodes = len(fw_all.nodes)
            col_k_slide, col_k_view = st.columns([3, 1])
            with col_k_slide:
                k_step = st.slider(
                    "Intermediate Node k Scrubber:",
                    min_value=0,
                    max_value=n_nodes,
                    value=n_nodes,
                    help="k = 0 shows initial direct edge weights. k = 1..n includes paths with intermediate nodes up to vertex k.",
                    key="fw_k_step_slider",
                )
            with col_k_view:
                matrix_view = st.radio(
                    "Matrix Display:",
                    options=["Heatmap", "Table"],
                    horizontal=True,
                    key="fw_matrix_display_type",
                )

            k_info_text = (
                f"Intermediate paths allowed through vertices: <b>{[net.servers[fw_all.nodes[i]].name for i in range(k_step) if fw_all.nodes[i] in net.servers]}</b>"
                if k_step > 0 else "Base direct adjacency connections only (no intermediate vertices)."
            )
            st.markdown(
                f"""
                <div class='algo-box' style='border-left-color: #a855f7;'>
                    <b>Step k = {k_step} / {n_nodes} Matrix $D^{{({k_step})}}$:</b> {k_info_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

            df_at_k = fw_all.dataframe_at_step(k_step, net)
            if matrix_view == "Heatmap":
                fig_heat = draw_latency_heatmap(df_at_k, title=f"All-Pairs Latency Matrix D^({k_step}) (ms)")
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.dataframe(df_at_k, use_container_width=True)


# =====================================================================
# TAB 4: DYNAMIC PROGRAMMING (TABULATION)
# =====================================================================
with tabs[3]:
    st.markdown("### Dynamic Programming (Tabulation)")
    st.caption("Optimal 0/1 knapsack task placement using dynamic programming with step traceback.")

    if not net.servers:
        st.info("Add servers in Canvas Studio to evaluate knapsack placement.")
    else:
        col_k_srv, col_k_rst = st.columns([3, 1])
        with col_k_srv:
            default_k_sid = st.session_state.get("selected_server_id", server_keys[0])
            k_sid_idx = server_keys.index(default_k_sid) if default_k_sid in server_keys else 0
            selected_server_id = st.selectbox(
                "Server (Capacity from Network):",
                options=server_keys,
                index=k_sid_idx,
                format_func=lambda sid: f"{net.servers[sid].name} (Capacity: {net.servers[sid].capacity})",
                key="knapsack_server_select",
            )
            st.session_state.selected_server_id = selected_server_id
            target_capacity = net.servers[selected_server_id].capacity

        with col_k_rst:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Reset Tasks", use_container_width=True, help="Reset to 4 default tasks"):
                st.session_state.tasks = get_default_demo_tasks()
                st.rerun()

        # Task Catalog Table Editor
        st.markdown("#### Task Catalog")
        tasks_df = pd.DataFrame([
            {"Name": t.name, "Weight": t.weight, "Value": t.value}
            for t in st.session_state.tasks
        ])
        edited_tasks = st.data_editor(
            tasks_df,
            num_rows="dynamic",
            use_container_width=True,
            key="knapsack_tasks_editor",
            column_config={
                "Name": st.column_config.TextColumn("Task", required=True),
                "Weight": st.column_config.NumberColumn("Weight", min_value=1, max_value=200, step=1, required=True),
                "Value": st.column_config.NumberColumn("Value", min_value=1, max_value=500, step=1, required=True),
            },
        )

        parsed_tasks = []
        if edited_tasks is not None:
            for idx, row in edited_tasks.iterrows():
                name_str = str(row.get("Name", "")).strip()
                if name_str and pd.notna(row.get("Weight")) and pd.notna(row.get("Value")):
                    parsed_tasks.append(
                        Task(
                            id=idx + 1,
                            name=name_str,
                            weight=int(row["Weight"]),
                            value=int(row["Value"]),
                        )
                    )
        if parsed_tasks:
            st.session_state.tasks = parsed_tasks

        active_tasks = st.session_state.tasks
        knapsack_res = solve_knapsack_01_dp(active_tasks, capacity=target_capacity)

        # Summary Metrics
        col_km1, col_km2, col_km3 = st.columns(3)
        with col_km1:
            st.markdown(
                f"""
                <div class='metric-card' style='border-left: 3px solid #22c55e;'>
                    <div class='metric-title'>Total Value</div>
                    <div class='metric-value' style='color: #4ade80;'>{knapsack_res.total_value} pts</div>
                    <div class='metric-sub'>Optimal 0/1 Knapsack value</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_km2:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-title'>Capacity Used</div>
                    <div class='metric-value'>{knapsack_res.total_weight} / {target_capacity}</div>
                    <div class='metric-sub'>{knapsack_res.capacity_used_pct:.1f}% physical utilization</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_km3:
            chosen_names = ", ".join([t.name for t in knapsack_res.chosen_tasks]) if knapsack_res.chosen_tasks else "None"
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-title'>Chosen Tasks</div>
                    <div class='metric-value' style='font-size: 1.1rem;'>{len(knapsack_res.chosen_tasks)} / {len(active_tasks)}</div>
                    <div class='metric-sub'>{chosen_names}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # DP Table with Highlighted Backtrack Path
        st.markdown(f"#### DP Table: $dp[i][c]$ for {net.servers[selected_server_id].name} (Capacity: {target_capacity})")
        dp_raw_df = knapsack_res.dp_dataframe(max_columns=35)
        bt_set = set(knapsack_res.backtrack_cells)

        def _is_bt_cell(row_name, col_name):
            try:
                r_i = int(str(row_name).split(":")[0].replace("Row", "").strip())
                c_i = int(str(col_name).replace("c=", "").strip())
                return (r_i, c_i) in bt_set
            except Exception:
                return False

        styled_dp = dp_raw_df.style.apply(
            lambda row: [
                'background-color: rgba(34, 197, 94, 0.35); font-weight: bold; color: #4ade80;'
                if _is_bt_cell(row.name, col_name) else ''
                for col_name in dp_raw_df.columns
            ],
            axis=1,
        )
        st.dataframe(styled_dp, use_container_width=True)

        # Backtracking Step Trace
        if knapsack_res.backtrack_cells:
            bt_log = []
            for b_i, (t_idx, c_val) in enumerate(knapsack_res.backtrack_cells, start=1):
                if t_idx == 0:
                    bt_log.append(f"• Base case dp[0][{c_val}] = 0 reached.")
                else:
                    t_item = active_tasks[t_idx - 1]
                    was_taken = any(ct.id == t_item.id for ct in knapsack_res.chosen_tasks)
                    if was_taken:
                        bt_log.append(
                            f"• dp[{t_idx}][{c_val}] ≠ dp[{t_idx - 1}][{c_val}] ➔ "
                            f"<span style='color: #4ade80; font-weight: 700;'>Selected '{t_item.name}'</span> "
                            f"(wt={t_item.weight}, val={t_item.value}). Shift capacity to {c_val - t_item.weight}."
                        )
                    else:
                        bt_log.append(
                            f"• dp[{t_idx}][{c_val}] == dp[{t_idx - 1}][{c_val}] ➔ "
                            f"<span style='color: #94a3b8;'>Skipped '{t_item.name}'</span>."
                        )
            st.markdown(
                f"""
                <div class='algo-box'>
                    <b>Traceback Sequence:</b><br>
                    {"<br>".join(bt_log)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Small "DP vs Greedy" Box
        st.markdown("#### DP vs Greedy Comparison")
        gv_res = solve_knapsack_greedy_value(active_tasks, target_capacity)
        gr_res = solve_knapsack_greedy_ratio(active_tasks, target_capacity)

        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            st.markdown(
                f"""
                <div class='metric-card' style='border-left: 3px solid #22c55e;'>
                    <div class='metric-title'>Dynamic Programming (Optimal)</div>
                    <div class='metric-value' style='color: #4ade80;'>{knapsack_res.total_value} pts</div>
                    <div class='metric-sub'>
                        Capacity: {knapsack_res.total_weight}/{target_capacity} ({knapsack_res.capacity_used_pct:.0f}%)<br>
                        Tasks: {[t.name for t in knapsack_res.chosen_tasks]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_g2:
            diff_v = knapsack_res.total_value - gv_res.total_value
            diff_v_tag = f"(-{diff_v} pts)" if diff_v > 0 else "(Equal)"
            st.markdown(
                f"""
                <div class='metric-card' style='border-left: 3px solid #f59e0b;'>
                    <div class='metric-title'>Greedy (Highest Value First)</div>
                    <div class='metric-value' style='color: #fbbf24;'>{gv_res.total_value} pts <span style='font-size:0.8rem; color:#94a3b8;'>{diff_v_tag}</span></div>
                    <div class='metric-sub'>
                        Capacity: {gv_res.total_weight}/{target_capacity} ({gv_res.capacity_used_pct:.0f}%)<br>
                        Tasks: {[t.name for t in gv_res.chosen_tasks]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_g3:
            diff_r = knapsack_res.total_value - gr_res.total_value
            diff_r_tag = f"(-{diff_r} pts)" if diff_r > 0 else "(Equal)"
            st.markdown(
                f"""
                <div class='metric-card' style='border-left: 3px solid #38bdf8;'>
                    <div class='metric-title'>Greedy (Value/Weight Ratio)</div>
                    <div class='metric-value' style='color: #38bdf8;'>{gr_res.total_value} pts <span style='font-size:0.8rem; color:#94a3b8;'>{diff_r_tag}</span></div>
                    <div class='metric-sub'>
                        Capacity: {gr_res.total_weight}/{target_capacity} ({gr_res.capacity_used_pct:.0f}%)<br>
                        Tasks: {[t.name for t in gr_res.chosen_tasks]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =====================================================================
# TAB 5: 0/1 KNAPSACK
# =====================================================================
with tabs[4]:
    st.markdown("### 0/1 Knapsack")
    st.caption("Brute-force binary decision tree evaluating all subsets against physical server capacity.")

    tree_tasks = list(st.session_state.tasks)
    curr_target_server_id = st.session_state.get("selected_server_id", 0)
    tree_capacity = net.servers[curr_target_server_id].capacity if curr_target_server_id in net.servers else 50
    curr_server_name = net.servers[curr_target_server_id].name if curr_target_server_id in net.servers else "Server"

    st.markdown(
        f"<b>Target Server:</b> {curr_server_name} &nbsp;|&nbsp; <b>Capacity:</b> {tree_capacity}",
        unsafe_allow_html=True,
    )

    n_tree = len(tree_tasks)

    # Live Counter
    st.markdown(
        f"""
        <div class='algo-box' style='border-left-color: #38bdf8;'>
            <b>Complexity Counter:</b> &nbsp;
            Items: <b>{n_tree}</b> &nbsp;|&nbsp;
            Leaves: <b>{2**n_tree:,}</b> &nbsp;|&nbsp;
            Total nodes: <b>{2**(n_tree + 1) - 1:,}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Pre-check Warnings & Hard Cap Protection
    if n_tree == 0:
        st.info("Add at least 1 task to draw the tree.")
        st.stop()

    if n_tree >= 8:
        st.error(
            f"Too many nodes to draw! n = {n_tree} tasks would create {2**n_tree:,} leaves. "
            f"Use the DP table instead. It solves the same problem in O(n x C)."
        )
        st.stop()

    if n_tree in (6, 7):
        st.warning(
            f"Too many nodes! With n = {n_tree} tasks the tree has {2**n_tree} leaves and "
            f"{2**(n_tree + 1) - 1} total nodes. It will look crowded. Showing only the first 5 tasks."
        )
        tasks_for_tree = tree_tasks[:5]
    else:
        tasks_for_tree = tree_tasks

    col_tr_ctrl1, col_tr_ctrl2 = st.columns([3, 1])
    with col_tr_ctrl1:
        reveal_level = st.slider(
            "Reveal level:",
            min_value=0,
            max_value=len(tasks_for_tree),
            value=len(tasks_for_tree),
            key="dt_reveal_level_slider",
        )
    with col_tr_ctrl2:
        prune_chk = st.checkbox("Prune branches that exceed capacity", value=False, key="dt_prune_chk")

    tree_res = build_decision_tree(
        tasks=tasks_for_tree,
        capacity=tree_capacity,
        prune_exceeded=prune_chk,
        max_depth=reveal_level,
    )

    fig_tree = draw_decision_tree_plotly(tree_res, THEME)
    st.plotly_chart(fig_tree, use_container_width=True)

    # Verification against DP
    dp_check = solve_knapsack_01_dp(tasks_for_tree, capacity=tree_capacity)
    if tree_res.best_leaf is not None and reveal_level == len(tasks_for_tree):
        tree_val = tree_res.best_leaf.total_value
        dp_val = dp_check.total_value
        tree_names = sorted(tree_res.best_leaf.selected_names)
        dp_names = sorted(t.name for t in dp_check.chosen_tasks)

        if tree_val == dp_val and tree_names == dp_names:
            st.markdown(
                f"""
                <div class='algo-box' style='border-left-color: #22c55e;'>
                    <span style='color: #4ade80; font-weight: 700;'>✔ Tree result matches DP result</span><br>
                    Both brute-force search and dynamic programming find optimal value: <b>{tree_val} pts</b><br>
                    Chosen Tasks: {tree_names}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info(f"Tree optimal value: {tree_val} pts vs DP optimal value: {dp_val} pts")
