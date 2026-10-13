"""
graph_utils.py - Data models and utilities for Cloud Task Placement Simulator.

Defines Server, Link, Client, and CloudNetwork data structures.
Includes realistic sample network, connected random network generator,
and JSON/CSV import/export utilities.
"""

from __future__ import annotations
import json
import math
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd


@dataclass
class Server:
    """Represents a cloud server node."""
    id: int
    name: str
    capacity: int  # Compute/task capacity units
    x: float       # Coordinate for 2D network visualization
    y: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Link:
    """Represents a network link with latency/net cost in milliseconds. Can be directed (one-way) or undirected."""
    u: int
    v: int
    latency: float         # ms (net cost = latency minus bandwidth credit)
    directed: bool = False # If True: directed u -> v; If False: bidirectional u <-> v

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Client:
    """Represents a client / demand point attached to a server."""
    id: int
    name: str
    server_id: int  # The server to which this client connects
    demand: int     # Required compute/task capacity units

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CloudNetwork:
    """
    Manages the complete cloud topology:
    - servers: Dict[int, Server]
    - links: List[Link]
    - clients: List[Client]
    """

    def __init__(
        self,
        servers: Optional[Dict[int, Server]] = None,
        links: Optional[List[Link]] = None,
        clients: Optional[List[Client]] = None,
    ):
        self.servers: Dict[int, Server] = servers or {}
        self.links: List[Link] = links or []
        self.clients: List[Client] = clients or []

    @property
    def num_servers(self) -> int:
        return len(self.servers)

    @property
    def num_links(self) -> int:
        return len(self.links)

    @property
    def total_capacity(self) -> int:
        return sum(s.capacity for s in self.servers.values())

    @property
    def total_demand(self) -> int:
        return sum(c.demand for c in self.clients)

    def adjacency_list(self) -> Dict[int, List[Tuple[int, float]]]:
        """Returns adjacency list {node: [(neighbor, latency), ...]}. Respects directed links."""
        adj: Dict[int, List[Tuple[int, float]]] = {sid: [] for sid in self.servers}
        for link in self.links:
            if link.u in adj and link.v in adj:
                adj[link.u].append((link.v, link.latency))
                if not link.directed:
                    adj[link.v].append((link.u, link.latency))
        return adj

    def directed_edges(self) -> List[Tuple[int, int, float]]:
        """
        Returns all directed edges (u, v, latency) for Bellman-Ford and shortest path relaxations.
        Undirected links produce two directed edges (u, v) and (v, u).
        Directed links produce one directed edge (u, v).
        """
        edges: List[Tuple[int, int, float]] = []
        for link in self.links:
            if link.u in self.servers and link.v in self.servers:
                edges.append((link.u, link.v, link.latency))
                if not link.directed:
                    edges.append((link.v, link.u, link.latency))
        return edges

    def unique_edges(self) -> List[Tuple[int, int, float]]:
        """
        Returns unique undirected edges (u, v, latency) where u < v for MST algorithms.
        Treats all links as undirected for MST (MST requires undirected graph).
        """
        edges: List[Tuple[int, int, float]] = []
        seen = set()
        for link in self.links:
            u, v = min(link.u, link.v), max(link.u, link.v)
            if (u, v) not in seen and u in self.servers and v in self.servers:
                seen.add((u, v))
                edges.append((u, v, link.latency))
        return edges

    def is_connected(self) -> bool:
        """
        Checks if the network graph is connected in the underlying undirected sense using BFS.
        Ensures servers form a single connected component for spanning tree / reachability.
        """
        if not self.servers:
            return False
        nodes = list(self.servers.keys())
        start_node = nodes[0]
        # Build undirected graph adjacency for structural connectivity
        undir_adj: Dict[int, List[int]] = {sid: [] for sid in self.servers}
        for link in self.links:
            if link.u in undir_adj and link.v in undir_adj:
                undir_adj[link.u].append(link.v)
                undir_adj[link.v].append(link.u)

        visited = set([start_node])
        queue = [start_node]
        while queue:
            curr = queue.pop(0)
            for neighbor in undir_adj.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return len(visited) == len(self.servers)

    def connected_components(self) -> List[set]:
        """Finds all connected components in the server network (undirected sense)."""
        undir_adj: Dict[int, List[int]] = {sid: [] for sid in self.servers}
        for link in self.links:
            if link.u in undir_adj and link.v in undir_adj:
                undir_adj[link.u].append(link.v)
                undir_adj[link.v].append(link.u)

        visited = set()
        components = []
        for sid in self.servers:
            if sid not in visited:
                comp = set()
                q = [sid]
                visited.add(sid)
                while q:
                    u = q.pop(0)
                    comp.add(u)
                    for v in undir_adj.get(u, []):
                        if v not in visited:
                            visited.add(v)
                            q.append(v)
                components.append(comp)
        return components

    def has_negative_weights(self) -> bool:
        """Returns True if any link has negative latency."""
        return any(link.latency < 0 for link in self.links)

    def validate(self, allow_negative_weights: bool = False, k: Optional[int] = None) -> Tuple[bool, List[str]]:
        """
        Validates the network for simulator requirements.
        Enforces negative weight rules:
        - If allow_negative_weights is False: reject negative weights.
        - Negative weights allowed ONLY on directed edges.
        - Undirected negative edge rejected: 'A negative undirected edge is an automatic negative cycle. Make it one-way.'
        Returns (is_valid, error_messages).
        """
        errors: List[str] = []
        if self.num_servers == 0:
            errors.append("Network contains 0 servers.")
            return False, errors

        # Negative weight validation
        for link in self.links:
            if link.latency < 0:
                u_name = self.servers[link.u].name if link.u in self.servers else link.u
                v_name = self.servers[link.v].name if link.v in self.servers else link.v
                if not allow_negative_weights:
                    errors.append(
                        f"Negative weight ({link.latency} ms) on link {u_name} {'→' if link.directed else '↔'} {v_name} is rejected. "
                        f"Enable 'Allow negative weights' to permit negative net costs."
                    )
                elif not link.directed:
                    errors.append(
                        f"Link {u_name} ↔ {v_name} has negative weight ({link.latency} ms). "
                        f"A negative undirected edge is an automatic negative cycle. Make it one-way."
                    )

        if not self.is_connected():
            comps = self.connected_components()
            errors.append(f"Network is partitioned into {len(comps)} disconnected components. Routing requires a connected graph.")

        if self.total_capacity < self.total_demand:
            errors.append(
                f"Total server capacity ({self.total_capacity} units) is less than total client demand ({self.total_demand} units)."
            )

        if k is not None:
            if k <= 0:
                errors.append("Target VM host limit (k) must be greater than 0.")
            if k > self.num_servers:
                errors.append(f"Target VM host limit k={k} exceeds the number of servers (n={self.num_servers}).")

        # Check client attachments
        for c in self.clients:
            if c.server_id not in self.servers:
                errors.append(f"Client '{c.name}' is attached to invalid server ID {c.server_id}.")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Serializes network to dictionary."""
        return {
            "servers": [s.to_dict() for s in self.servers.values()],
            "links": [l.to_dict() for l in self.links],
            "clients": [c.to_dict() for c in self.clients],
        }

    def to_json(self) -> str:
        """Serializes network to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CloudNetwork:
        """Reconstructs CloudNetwork from dictionary."""
        servers = {
            s["id"]: Server(
                id=int(s["id"]),
                name=str(s["name"]),
                capacity=int(s["capacity"]),
                x=float(s.get("x", 0.0)),
                y=float(s.get("y", 0.0)),
            )
            for s in data.get("servers", [])
        }
        links = [
            Link(
                u=int(l["u"]),
                v=int(l["v"]),
                latency=float(l["latency"]),
                directed=bool(l.get("directed", False)),
            )
            for l in data.get("links", [])
        ]
        clients = [
            Client(
                id=int(c["id"]),
                name=str(c["name"]),
                server_id=int(c["server_id"]),
                demand=int(c["demand"]),
            )
            for c in data.get("clients", [])
        ]
        return cls(servers=servers, links=links, clients=clients)

    @classmethod
    def from_json(cls, json_str: str) -> CloudNetwork:
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_csv_dfs(
        cls,
        servers_df: pd.DataFrame,
        links_df: pd.DataFrame,
        clients_df: pd.DataFrame,
    ) -> CloudNetwork:
        """Creates CloudNetwork from uploaded pandas DataFrames."""
        servers = {}
        for _, row in servers_df.iterrows():
            sid = int(row["id"])
            servers[sid] = Server(
                id=sid,
                name=str(row.get("name", f"Server-{sid}")),
                capacity=int(row.get("capacity", 50)),
                x=float(row.get("x", 0.0)),
                y=float(row.get("y", 0.0)),
            )

        links = []
        for _, row in links_df.iterrows():
            links.append(
                Link(
                    u=int(row["u"]),
                    v=int(row["v"]),
                    latency=float(row.get("latency", 20.0)),
                    directed=bool(row.get("directed", False)),
                )
            )

        clients = []
        for _, row in clients_df.iterrows():
            clients.append(
                Client(
                    id=int(row["id"]),
                    name=str(row.get("name", f"Client-{row['id']}")),
                    server_id=int(row["server_id"]),
                    demand=int(row.get("demand", 10)),
                )
            )

        return cls(servers=servers, links=links, clients=clients)


def create_sample_network() -> CloudNetwork:
    """
    Builds the built-in 10-city sample network with realistic latencies,
    capacities, and clients.
    Cities: Mumbai, Delhi, Hyderabad, Bangalore, Chennai, Kolkata,
            Singapore, Dubai, London, Frankfurt.
    """
    servers = {
        0: Server(id=0, name="Mumbai", capacity=50, x=340.0, y=360.0),
        1: Server(id=1, name="Delhi", capacity=45, x=330.0, y=200.0),
        2: Server(id=2, name="Hyderabad", capacity=40, x=380.0, y=420.0),
        3: Server(id=3, name="Bangalore", capacity=55, x=370.0, y=500.0),
        4: Server(id=4, name="Chennai", capacity=40, x=420.0, y=520.0),
        5: Server(id=5, name="Kolkata", capacity=35, x=530.0, y=310.0),
        6: Server(id=6, name="Singapore", capacity=60, x=670.0, y=560.0),
        7: Server(id=7, name="Dubai", capacity=50, x=190.0, y=320.0),
        8: Server(id=8, name="London", capacity=65, x=60.0, y=100.0),
        9: Server(id=9, name="Frankfurt", capacity=60, x=140.0, y=120.0),
    }

    # Realistic inter-city network links with latency in ms:
    # Domestic Indian links: 7 - 25 ms
    # Middle East / Asia links: 30 - 45 ms
    # Transcontinental links to Europe: 80 - 150 ms
    raw_links = [
        (0, 1, 18.0),   # Mumbai - Delhi
        (0, 2, 14.0),   # Mumbai - Hyderabad
        (0, 3, 19.0),   # Mumbai - Bangalore
        (1, 5, 24.0),   # Delhi - Kolkata
        (2, 3, 11.0),   # Hyderabad - Bangalore
        (3, 4, 7.0),    # Bangalore - Chennai
        (2, 5, 22.0),   # Hyderabad - Kolkata
        (4, 5, 27.0),   # Chennai - Kolkata
        (4, 6, 38.0),   # Chennai - Singapore
        (5, 6, 44.0),   # Kolkata - Singapore
        (0, 7, 32.0),   # Mumbai - Dubai
        (1, 7, 39.0),   # Delhi - Dubai
        (7, 9, 84.0),   # Dubai - Frankfurt
        (8, 9, 12.0),   # London - Frankfurt
        (7, 8, 92.0),   # Dubai - London
        (0, 9, 105.0),  # Mumbai - Frankfurt
        (6, 8, 145.0),  # Singapore - London
    ]

    links = [Link(u=u, v=v, latency=lat) for u, v, lat in raw_links]

    clients = [
        Client(id=0, name="FinTech Core", server_id=0, demand=25),       # Mumbai
        Client(id=1, name="Gov Public Cloud", server_id=1, demand=20),   # Delhi
        Client(id=2, name="E-Comm Logistics", server_id=3, demand=30),   # Bangalore
        Client(id=3, name="APAC Stream CDN", server_id=6, demand=28),    # Singapore
        Client(id=4, name="EU Quant Trading", server_id=8, demand=22),   # London
        Client(id=5, name="Health AI Compute", server_id=2, demand=15),  # Hyderabad
    ]

    return CloudNetwork(servers=servers, links=links, clients=clients)


def generate_random_network(
    n: int = 10,
    edge_density: float = 0.35,
    seed: int = 42,
) -> CloudNetwork:
    """
    Generates a connected random network with n servers and realistic parameters.
    Guarantees that the graph is connected by first constructing a random spanning tree,
    then sampling additional edges according to edge_density.
    """
    rng = random.Random(seed)

    city_names = [
        "Mumbai", "Delhi", "Hyderabad", "Bangalore", "Chennai", "Kolkata",
        "Singapore", "Dubai", "London", "Frankfurt", "Tokyo", "Sydney",
        "New York", "San Francisco", "São Paulo", "Toronto", "Seoul", "Paris"
    ]

    # Assign positions around a 2D plane
    servers: Dict[int, Server] = {}
    positions: List[Tuple[float, float]] = []

    for i in range(n):
        name = city_names[i] if i < len(city_names) else f"CloudCity-{i+1}"
        cap = rng.randint(35, 75)
        # Use circular/jittered distribution for clean visual layout
        angle = 2.0 * math.pi * i / n
        radius = rng.uniform(200.0, 320.0)
        cx, cy = 400.0, 350.0
        x = cx + radius * math.cos(angle) + rng.uniform(-30.0, 30.0)
        y = cy + radius * math.sin(angle) + rng.uniform(-30.0, 30.0)
        positions.append((x, y))
        servers[i] = Server(id=i, name=name, capacity=cap, x=round(x, 1), y=round(y, 1))

    # Guarantee connectivity via randomized spanning tree
    tree_edges: List[Tuple[int, int]] = []
    connected_nodes = [0]
    unconnected_nodes = list(range(1, n))
    rng.shuffle(unconnected_nodes)

    while unconnected_nodes:
        u = rng.choice(connected_nodes)
        v = unconnected_nodes.pop()
        tree_edges.append((min(u, v), max(u, v)))
        connected_nodes.append(v)

    edge_set = set(tree_edges)

    # Add extra random edges based on edge_density
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edge_set:
                if rng.random() < edge_density:
                    edge_set.add((i, j))

    # Calculate latency based on Euclidean distance with jitter
    links: List[Link] = []
    for u, v in sorted(edge_set):
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        dist = math.hypot(x2 - x1, y2 - y1)
        # Map distance to realistic latency (approx 5ms to 120ms)
        latency = round(dist * 0.18 + rng.uniform(2.0, 15.0), 1)
        latency = max(5.0, latency)
        links.append(Link(u=u, v=v, latency=latency))

    # Generate 5-6 clients
    num_clients = max(4, min(7, n - 2))
    client_servers = rng.sample(list(range(n)), num_clients)
    clients: List[Client] = []
    for cid, s_target in enumerate(client_servers):
        demand = rng.randint(15, 30)
        clients.append(
            Client(
                id=cid,
                name=f"Client-{chr(65 + cid)} ({servers[s_target].name})",
                server_id=s_target,
                demand=demand,
            )
        )

    return CloudNetwork(servers=servers, links=links, clients=clients)


def create_blank_network() -> CloudNetwork:
    """Creates a blank network with zero servers, links, or clients for manual building."""
    return CloudNetwork(servers={}, links=[], clients=[])


def create_triangle_network() -> CloudNetwork:
    """Creates a simple 3-node triangular network for intuitive educational demos."""
    servers = {
        0: Server(id=0, name="Server-A", capacity=60, x=400.0, y=180.0),
        1: Server(id=1, name="Server-B", capacity=50, x=240.0, y=460.0),
        2: Server(id=2, name="Server-C", capacity=45, x=560.0, y=460.0),
    }
    links = [
        Link(u=0, v=1, latency=14.0),
        Link(u=1, v=2, latency=9.0),
        Link(u=0, v=2, latency=22.0),
    ]
    clients = [
        Client(id=0, name="Client-Alpha", server_id=0, demand=20),
        Client(id=1, name="Client-Beta", server_id=2, demand=25),
    ]
    return CloudNetwork(servers=servers, links=links, clients=clients)


def create_ring_network() -> CloudNetwork:
    """Creates a simple 5-node ring network with cross links."""
    servers = {
        0: Server(id=0, name="Node-1", capacity=50, x=400.0, y=160.0),
        1: Server(id=1, name="Node-2", capacity=55, x=580.0, y=280.0),
        2: Server(id=2, name="Node-3", capacity=45, x=510.0, y=500.0),
        3: Server(id=3, name="Node-4", capacity=40, x=290.0, y=500.0),
        4: Server(id=4, name="Node-5", capacity=60, x=220.0, y=280.0),
    }
    links = [
        Link(u=0, v=1, latency=15.0),
        Link(u=1, v=2, latency=18.0),
        Link(u=2, v=3, latency=12.0),
        Link(u=3, v=4, latency=16.0),
        Link(u=4, v=0, latency=20.0),
        Link(u=0, v=2, latency=35.0),
        Link(u=1, v=4, latency=28.0),
    ]
    clients = [
        Client(id=0, name="Client-Web", server_id=0, demand=25),
        Client(id=1, name="Client-DB", server_id=2, demand=30),
        Client(id=2, name="Client-App", server_id=4, demand=18),
    ]
    return CloudNetwork(servers=servers, links=links, clients=clients)


def create_dijkstra_vs_bellman_network() -> CloudNetwork:
    """
    Preset 1: Counterexample graph where Dijkstra fails and Bellman-Ford succeeds.
    Directed edges:
      S -> A = 4
      S -> B = 5
      B -> A = -3
    Origin: S (ID 0), Destination: A (ID 1).
    Expected:
      - Dijkstra: 4 (wrong, path S -> A; settles A prematurely with greedy distance 4)
      - Bellman-Ford: 2 (correct, path S -> B -> A; cost 5 + (-3) = 2)
    """
    servers = {
        0: Server(id=0, name="S", capacity=50, x=220.0, y=320.0),
        1: Server(id=1, name="A", capacity=60, x=540.0, y=200.0),
        2: Server(id=2, name="B", capacity=45, x=380.0, y=480.0),
    }
    links = [
        Link(u=0, v=1, latency=4.0, directed=True),   # S -> A = 4
        Link(u=0, v=2, latency=5.0, directed=True),   # S -> B = 5
        Link(u=2, v=1, latency=-3.0, directed=True),  # B -> A = -3
    ]
    clients = [
        Client(id=0, name="Client-App", server_id=1, demand=15),
    ]
    return CloudNetwork(servers=servers, links=links, clients=clients)


def create_negative_cycle_network() -> CloudNetwork:
    """
    Preset 2: Negative Cycle Demonstration.
    Directed edges:
      S -> A = 4
      S -> B = 5
      B -> A = -3
      A -> B = 1
    Cycle: B -> A -> B has total negative cost (-3) + 1 = -2 < 0.
    Expected:
      - Bellman-Ford reports 'NEGATIVE CYCLE DETECTED' with cycle containing A and B.
      - Shortest path distance to affected nodes is -infinity.
    """
    servers = {
        0: Server(id=0, name="S", capacity=50, x=220.0, y=320.0),
        1: Server(id=1, name="A", capacity=60, x=540.0, y=200.0),
        2: Server(id=2, name="B", capacity=45, x=380.0, y=480.0),
    }
    links = [
        Link(u=0, v=1, latency=4.0, directed=True),   # S -> A = 4
        Link(u=0, v=2, latency=5.0, directed=True),   # S -> B = 5
        Link(u=2, v=1, latency=-3.0, directed=True),  # B -> A = -3
        Link(u=1, v=2, latency=1.0, directed=True),   # A -> B = 1
    ]
    clients = [
        Client(id=0, name="Client-App", server_id=1, demand=15),
    ]
    return CloudNetwork(servers=servers, links=links, clients=clients)


def create_normal_network_5() -> CloudNetwork:
    """
    Normal 5-node cloud network for college demonstration.
    Servers: S0 (cap 50), S1 (cap 40), S2 (cap 60), S3 (cap 45), S4 (cap 50).
    Undirected links with positive latency, suitable for Kruskal MST and standard Dijkstra.
    """
    servers = {
        0: Server(id=0, name="S0", capacity=50, x=240.0, y=300.0),
        1: Server(id=1, name="S1", capacity=40, x=380.0, y=160.0),
        2: Server(id=2, name="S2", capacity=60, x=540.0, y=360.0),
        3: Server(id=3, name="S3", capacity=45, x=400.0, y=460.0),
        4: Server(id=4, name="S4", capacity=50, x=640.0, y=220.0),
    }
    links = [
        Link(u=0, v=1, latency=15.0, directed=False),
        Link(u=0, v=3, latency=12.0, directed=False),
        Link(u=1, v=2, latency=25.0, directed=False),
        Link(u=1, v=4, latency=30.0, directed=False),
        Link(u=2, v=3, latency=10.0, directed=False),
        Link(u=2, v=4, latency=18.0, directed=False),
    ]
    clients = [
        Client(id=0, name="Client-1", server_id=0, demand=15),
        Client(id=1, name="Client-2", server_id=2, demand=20),
    ]
    return CloudNetwork(servers=servers, links=links, clients=clients)


