# Cloud Task Placement Simulator ☁️

A college algorithms project for **Design and Analysis of Algorithms (DAA PBL)** demonstrating three fundamental algorithmic paradigms on a simulated cloud network:
1. **Minimum Spanning Tree (MST)** using **Kruskal's Algorithm** (Union-Find) and **Prim's Algorithm** (Min-Heap) for backbone topology.
2. **Shortest-Path Routing** using **Dijkstra's Algorithm** (min-heap), **Bellman-Ford Algorithm** (edge relaxation passes), and **Floyd-Warshall Algorithm** (all-pairs dynamic programming).
3. **Task Placement Optimization** using **Covering Knapsack Dynamic Programming** to select VM host servers minimizing total routing delay from a user entry node while meeting total task demand $D$.

> **Visual Theme:** Sleek Obsidian / Shades of Black dark aesthetic with deep dark crimson and red accents for high contrast and modern UI presentation.

---

## 📋 Problem Statement

Cloud service providers need to select a subset of servers to host virtual machines (VMs) for an incoming cloud workload:
- **Servers ($V$):** Data centers in an undirected graph with capacity constraints $C_v$ and geographic coordinates.
- **Links ($E$):** Fiber links with propagation latencies $L_{u,v}$ in milliseconds (ms).
- **Task Demand ($D$):** Total compute demand units required by incoming cloud workloads.
- **Backbone Network:** Minimal latency spanning tree connecting all servers without cycles.
- **Shortest Paths:** Optimal routing delays from user entry node to candidate host servers via Dijkstra's algorithm.
- **Placement Optimization:** Choosing a subset of servers (optionally at most $k$) whose total capacity $\sum_{s \in S} C_s \ge D$ with minimum total routing delay $\sum_{s \in S} \text{delay}_s$.

---

## 🛠️ Algorithms Implemented (Manual from Scratch)

> **Note:** All core algorithms are manually implemented without relying on NetworkX's built-in MST or shortest-path functions. NetworkX is used exclusively for graph structure layout and in unit tests to verify algorithmic correctness.

| Algorithm | Paradigm | File | Time Complexity | Space Complexity | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Kruskal's Algorithm** | Greedy + DSU | `mst.py` | $\mathcal{O}(\|E\| \log \|E\| + \|E\| \alpha(\|V\|))$ | $\mathcal{O}(\|V\| + \|E\|)$ | Minimal latency backbone via edge sorting & DSU |
| **Prim's Algorithm** | Greedy + Min-Heap | `mst.py` | $\mathcal{O}(\|E\| \log \|V\|)$ | $\mathcal{O}(\|V\| + \|E\|)$ | Minimal latency backbone via growing tree frontier |
| **Dijkstra's Algorithm** | Greedy + Min-Heap | `shortest_path.py` | $\mathcal{O}((\|V\| + \|E\|) \log \|V\|)$ | $\mathcal{O}(\|V\| + \|E\|)$ | Single-source shortest path routing |
| **Bellman-Ford** | Dynamic Programming | `shortest_path.py` | $\mathcal{O}(\|V\| \cdot \|E\|)$ | $\mathcal{O}(\|V\|)$ | Single-source shortest path via $|V|-1$ relaxations |
| **Floyd-Warshall** | Dynamic Programming | `shortest_path.py` | $\mathcal{O}(\|V\|^3)$ | $\mathcal{O}(\|V\|^2)$ | All-pairs latency matrix & path reconstruction |
| **Knapsack DP** | Dynamic Programming | `placement_dp.py` | $\mathcal{O}(n \times D)$ | $\mathcal{O}(n \times D)$ | Optimal VM host server placement under demand $D$ |
| **Greedy Placement** | Greedy Baseline | `baselines.py` | $\mathcal{O}(n \log n)$ | $\mathcal{O}(n)$ | Cost-efficiency ratio sorting baseline |
| **Random Placement** | Monte Carlo Baseline | `baselines.py` | $\mathcal{O}(\text{trials} \cdot n)$ | $\mathcal{O}(n)$ | Empirical random subset baseline |

---

## 📁 Project File Structure

```text
DAA PBL/
├── app.py                  # Main Streamlit web application with 7 interactive tabs
├── graph_utils.py          # Data models (Server, Link, Client, CloudNetwork) & generators
├── mst.py                  # Manual Kruskal (Union-Find) and Prim (Min-Heap)
├── shortest_path.py        # Manual Dijkstra, Bellman-Ford & Floyd-Warshall
├── placement_dp.py         # Knapsack DP solver & Brute-force validator
├── baselines.py            # Greedy heuristic & Random baseline comparison runners
├── visualization.py        # Dark theme Plotly visualizers (shades of black & dark red)
├── requirements.txt        # Python dependency manifest
├── tests/
│   └── test_algorithms.py  # 22 automated pytest tests against NetworkX and Brute-force
├── .streamlit/
│   └── config.toml         # Custom shades-of-black & dark red Streamlit configuration
└── README.md               # Project documentation and demo script
```

---

## 🚀 Quick Start & Launch

### Option A: One-Click Launch (macOS Finder) 🖱️
Simply **double-click** the file `run.command` in the project folder!
- macOS will automatically open Terminal.app.
- It verifies Python, environment dependencies, runs the 22-test algorithmic verification suite, and launches the Streamlit web application.
- Your browser will open to `http://localhost:8501`.

*(On Windows, double-click `run.bat`; on Linux, run `./run.sh`).*

---

## 🖥️ Application Features & Tabs

1. **🌐 Network:** Interactive physical cloud topology, server capacities, client demands, and connectivity status.
2. **🌲 MST Backbone:** Kruskal's algorithm visualization with step-through slider tracking edge-by-edge decisions (ACCEPTED vs REJECTED cycle).
3. **⚡ Routing:** Dijkstra and Bellman-Ford route finders with animated edge relaxations, plus Floyd-Warshall all-pairs latency heatmap.
4. **🎯 Placement (DP):** Knapsack DP table $dp[i][c]$ displaying step-by-step state transitions, backtracking trace, and selected VM host servers.
5. **📊 Comparison:** Benchmark table and bar charts comparing Knapsack DP vs Greedy vs Random on total delay, allocated capacity, and runtime.
6. **💥 Failure Test:** Simulate a severed link or crashed server, re-evaluating topology resilience and dynamic rerouting impact.
7. **🎛️ Capacity Demo:** Live slider to throttle a chosen VM host server's capacity and watch the DP automatically shift placement to avoid overload.

---

## ⏱️ Suggested 5-Minute College Viva / Presentation Demo Script

### Minute 1: Introduction & Problem Context (Tab 1: Network)
- **Say:** *"Cloud service providers need to deploy incoming workloads across physical servers. Given a user entry node, each server has a shortest-path delay and a capacity constraint. Our goal is to satisfy total task demand $D$ while minimizing total routing delay."*
- **Show:** The built-in 10-city sample network (Mumbai, Delhi, Hyderabad, Bangalore, Chennai, Kolkata, Singapore, Dubai, London, Frankfurt). Point out server capacities and fiber latencies.

### Minute 2: Minimum Spanning Tree Backbone (Tab 2: MST Backbone)
- **Say:** *"First, we construct the backbone network using Kruskal's algorithm implemented from scratch with a Disjoint-Set Union (Union-Find) data structure featuring path compression and union-by-rank."*
- **Action:** Move the **Kruskal Step Slider**. Show how low-latency edges are accepted, and point out when an edge is **rejected** because both endpoints already share the same component root, preventing cycles in $\mathcal{O}(|E| \log |E|)$ time.

### Minute 3: Routing & Latency Matrices (Tab 3: Routing)
- **Say:** *"Next, for message routing, we implement Dijkstra's algorithm using a min-heap priority queue for single-source routing, and Floyd-Warshall's $\mathcal{O}(|V|^3)$ dynamic programming algorithm for the all-pairs latency matrix."*
- **Action:** Select an origin server and a destination server. Show the neon crimson highlighted shortest path with its exact hop sequence and latency. Scroll to the Floyd-Warshall heatmap.

### Minute 4: Task Placement with Knapsack DP (Tab 4: Placement)
- **Say:** *"The core optimization problem is selecting VM host servers to reach capacity $D$ with minimum delay from the user entry node. We formulate this as a Covering Knapsack Dynamic Program with recurrence $dp[i][c] = \min(dp[i-1][c], dp[i-1][\max(0, c - cap_i)] + delay_i)$, running in $\mathcal{O}(n \times D)$ time."*
- **Action:** Set Demand $D = 80$. Point out the glowing crimson VM host server nodes, the 2D DP table $dp[i][c]$, the backtracking trace, and the selected host servers.

### Minute 5: Algorithm Benchmark & Capacity Sensitivity (Tabs 5 & 7)
- **Say:** *"We benchmark our exact DP against two standard baselines: Greedy Heuristic (ratio sorting) and Random Baseline."*
- **Action 1 (Tab 5):** Show the comparison bar charts. Explain that DP achieves the provable global minimum total delay, while Greedy can overshoot.
- **Action 2 (Tab 7 - Live Capacity Demo):** Pick one of the selected VM host servers (e.g., Bangalore or Singapore) and drag the capacity slider down. Show how the algorithm dynamically reroutes demand and relocates VM host servers to avoid server overload!
- **Conclude:** *"This confirms that our simulation successfully combines Greedy, Graph Theory, and Dynamic Programming into an end-to-end cloud optimization system."*
