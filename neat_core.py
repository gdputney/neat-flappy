"""Core NEAT genome abstractions for Flappy Bird experiments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import random
from typing import Any


@dataclass
class InnovationTracker:
    """Tracks globally consistent innovation numbers for structural mutations."""

    next_innovation: int = 0
    connection_innovations: dict[tuple[int, int], int] = field(default_factory=dict)

    def get_connection_innovation(self, in_node: int, out_node: int) -> int:
        key = (int(in_node), int(out_node))
        if key not in self.connection_innovations:
            self.connection_innovations[key] = self.next_innovation
            self.next_innovation += 1
        return self.connection_innovations[key]


@dataclass
class Genome:
    """Represents an evolvable feedforward NEAT genome."""

    node_genes: list[dict[str, Any]] = field(default_factory=list)
    connection_genes: list[dict[str, Any]] = field(default_factory=list)
    fitness: float = 0.0

    def activate(self, inputs: list[float]) -> list[float]:
        """Run feedforward evaluation via topological ordering over enabled edges."""
        input_nodes = self._nodes_by_type("input")
        output_nodes = self._nodes_by_type("output")
        if len(inputs) != len(input_nodes):
            raise ValueError(f"Expected {len(input_nodes)} inputs, received {len(inputs)}")

        self._ensure_acyclic_enabled_connections()

        node_lookup = {int(node["id"]): node for node in self.node_genes}
        enabled_edges = [gene for gene in self.connection_genes if gene.get("enabled", True)]

        indegree = {int(node["id"]): 0 for node in self.node_genes}
        outgoing: dict[int, list[dict[str, Any]]] = {int(node["id"]): [] for node in self.node_genes}
        for edge in enabled_edges:
            in_node = int(edge["in_node"])
            out_node = int(edge["out_node"])
            indegree[out_node] = indegree.get(out_node, 0) + 1
            outgoing.setdefault(in_node, []).append(edge)

        queue = deque([node_id for node_id, degree in indegree.items() if degree == 0])
        topo_order: list[int] = []
        while queue:
            current = queue.popleft()
            topo_order.append(current)
            for edge in outgoing.get(current, []):
                out_node = int(edge["out_node"])
                indegree[out_node] -= 1
                if indegree[out_node] == 0:
                    queue.append(out_node)

        values = {int(node["id"]): 0.0 for node in self.node_genes}
        for input_value, node in zip(inputs, input_nodes):
            values[int(node["id"])] = float(input_value)

        incoming_map: dict[int, list[dict[str, Any]]] = {int(node["id"]): [] for node in self.node_genes}
        for edge in enabled_edges:
            incoming_map.setdefault(int(edge["out_node"]), []).append(edge)

        for node_id in topo_order:
            node = node_lookup[node_id]
            if node.get("type") == "input":
                continue

            weighted_sum = float(node.get("bias", 0.0))
            for edge in incoming_map.get(node_id, []):
                in_node = int(edge["in_node"])
                weighted_sum += float(edge.get("weight", 0.0)) * values[in_node]
            values[node_id] = self._sigmoid(weighted_sum)

        return [values[int(node["id"])] for node in output_nodes]

    def mutate(self, tracker: InnovationTracker | None = None) -> None:
        """Apply NEAT mutations: weights, add-node, add-connection, enable/disable toggle."""
        self._perturb_connection_weights()

        if random.random() < 0.05:
            self._toggle_connection_enabled()
        if random.random() < 0.2:
            self._add_connection_mutation(tracker)
        if random.random() < 0.1:
            self._add_node_mutation(tracker)

        self._ensure_acyclic_enabled_connections()

    def _perturb_connection_weights(self, sigma: float = 0.3, reset_chance: float = 0.1) -> None:
        for connection in self.connection_genes:
            if random.random() < reset_chance:
                connection["weight"] = random.uniform(-1.0, 1.0)
            else:
                connection["weight"] = float(connection.get("weight", 0.0)) + random.gauss(0.0, sigma)

    def _toggle_connection_enabled(self) -> None:
        if not self.connection_genes:
            return
        gene = random.choice(self.connection_genes)
        currently_enabled = bool(gene.get("enabled", True))
        if currently_enabled:
            gene["enabled"] = False
            return

        in_node = int(gene["in_node"])
        out_node = int(gene["out_node"])
        if self._has_enabled_path(out_node, in_node):
            return
        gene["enabled"] = True

    def _add_node_mutation(self, tracker: InnovationTracker | None = None) -> None:
        enabled_connections = [gene for gene in self.connection_genes if gene.get("enabled", True)]
        if not enabled_connections:
            return

        connection = random.choice(enabled_connections)
        connection["enabled"] = False

        in_node = int(connection["in_node"])
        out_node = int(connection["out_node"])
        original_weight = float(connection["weight"])

        new_node_id = self._next_node_id()
        self.node_genes.append({"id": new_node_id, "type": "hidden", "bias": 0.0})

        self.connection_genes.append(
            {
                "in_node": in_node,
                "out_node": new_node_id,
                "weight": 1.0,
                "enabled": True,
                "innovation": self._innovation_for(tracker, in_node, new_node_id),
            }
        )
        self.connection_genes.append(
            {
                "in_node": new_node_id,
                "out_node": out_node,
                "weight": original_weight,
                "enabled": True,
                "innovation": self._innovation_for(tracker, new_node_id, out_node),
            }
        )

    def _add_connection_mutation(
        self,
        tracker: InnovationTracker | None = None,
        max_attempts: int = 30,
    ) -> None:
        node_ids = [int(node["id"]) for node in self.node_genes]
        if len(node_ids) < 2:
            return

        depth = self._node_depths()

        for _ in range(max_attempts):
            in_node = random.choice(node_ids)
            out_node = random.choice(node_ids)
            if in_node == out_node:
                continue
            if self._node_type(out_node) == "input":
                continue
            if depth.get(in_node, 0) >= depth.get(out_node, 0):
                continue

            existing_gene = self._find_connection_gene(in_node, out_node)
            if existing_gene is not None:
                if not existing_gene.get("enabled", True) and not self._has_enabled_path(out_node, in_node):
                    existing_gene["enabled"] = True
                    existing_gene["weight"] = random.uniform(-1.0, 1.0)
                continue

            if self._has_enabled_path(out_node, in_node):
                continue

            self.connection_genes.append(
                {
                    "in_node": in_node,
                    "out_node": out_node,
                    "weight": random.uniform(-1.0, 1.0),
                    "enabled": True,
                    "innovation": self._innovation_for(tracker, in_node, out_node),
                }
            )
            return

    def crossover(self, other: "Genome") -> "Genome":
        if not isinstance(other, Genome):
            raise TypeError("Genome.crossover() requires another Genome")

        fitter, weaker = (self, other) if self.fitness >= other.fitness else (other, self)
        equal_fitness = self.fitness == other.fitness

        fitter_map = {int(g["innovation"]): g for g in fitter.connection_genes}
        weaker_map = {int(g["innovation"]): g for g in weaker.connection_genes}

        child_connections: list[dict[str, Any]] = []
        all_innovations = sorted(set(fitter_map) | set(weaker_map))
        for innovation in all_innovations:
            gene_a = fitter_map.get(innovation)
            gene_b = weaker_map.get(innovation)

            if gene_a is not None and gene_b is not None:
                chosen = random.choice([gene_a, gene_b]).copy()
                if (not gene_a.get("enabled", True) or not gene_b.get("enabled", True)) and random.random() < 0.75:
                    chosen["enabled"] = False
                child_connections.append(chosen)
            elif equal_fitness and (gene_a or gene_b):
                child_connections.append((gene_a or gene_b).copy())
            elif gene_a is not None:
                child_connections.append(gene_a.copy())

        node_ids = {int(node["id"]) for node in self.node_genes} | {int(node["id"]) for node in other.node_genes}
        node_lookup = {
            int(node["id"]): node
            for node in (self.node_genes + other.node_genes)
            if int(node["id"]) in node_ids
        }
        child_nodes = [node_lookup[node_id].copy() for node_id in sorted(node_ids)]

        child = Genome(node_genes=child_nodes, connection_genes=child_connections, fitness=0.0)
        child._ensure_acyclic_enabled_connections()
        return child

    def compatibility_distance(self, other: "Genome", c1: float = 1.0, c2: float = 1.0, c3: float = 0.4) -> float:
        """NEAT compatibility distance based on excess/disjoint genes and weight difference."""
        self_map = {int(g["innovation"]): g for g in self.connection_genes}
        other_map = {int(g["innovation"]): g for g in other.connection_genes}
        if not self_map and not other_map:
            return 0.0

        matching = set(self_map) & set(other_map)
        all_keys = set(self_map) | set(other_map)

        max_self = max(self_map) if self_map else -1
        max_other = max(other_map) if other_map else -1
        min_max = min(max_self, max_other)

        excess = sum(1 for k in all_keys if k > min_max)
        disjoint = len(all_keys) - len(matching) - excess

        avg_weight_diff = 0.0
        if matching:
            avg_weight_diff = sum(
                abs(float(self_map[k]["weight"]) - float(other_map[k]["weight"])) for k in matching
            ) / len(matching)

        n = max(len(self_map), len(other_map), 1)
        return (c1 * excess / n) + (c2 * disjoint / n) + (c3 * avg_weight_diff)

    def _find_connection_gene(self, in_node: int, out_node: int) -> dict[str, Any] | None:
        for gene in self.connection_genes:
            if int(gene["in_node"]) == in_node and int(gene["out_node"]) == out_node:
                return gene
        return None

    def _has_enabled_path(self, start_node: int, target_node: int) -> bool:
        if start_node == target_node:
            return True

        adjacency: dict[int, list[int]] = {}
        for gene in self.connection_genes:
            if not gene.get("enabled", True):
                continue
            src = int(gene["in_node"])
            dst = int(gene["out_node"])
            adjacency.setdefault(src, []).append(dst)

        visited: set[int] = set()
        queue = deque([start_node])
        while queue:
            current = queue.popleft()
            if current == target_node:
                return True
            if current in visited:
                continue
            visited.add(current)
            for nxt in adjacency.get(current, []):
                if nxt not in visited:
                    queue.append(nxt)
        return False

    def _node_depths(self) -> dict[int, int]:
        """Compute depth/layer from current enabled feedforward graph; fallback to type-based ordering."""
        node_ids = [int(node["id"]) for node in self.node_genes]
        indegree = {node_id: 0 for node_id in node_ids}
        outgoing: dict[int, list[int]] = {node_id: [] for node_id in node_ids}

        for gene in self.connection_genes:
            if not gene.get("enabled", True):
                continue
            src = int(gene["in_node"])
            dst = int(gene["out_node"])
            if self._node_type(dst) == "input":
                continue
            outgoing.setdefault(src, []).append(dst)
            indegree[dst] = indegree.get(dst, 0) + 1

        queue = deque([node_id for node_id, deg in indegree.items() if deg == 0])
        depth = {node_id: 0 for node_id in queue}
        seen = 0

        while queue:
            current = queue.popleft()
            seen += 1
            current_depth = depth.get(current, 0)
            for nxt in outgoing.get(current, []):
                depth[nxt] = max(depth.get(nxt, 0), current_depth + 1)
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if seen != len(node_ids):
            return self._topological_rank()

        for node_id in node_ids:
            depth.setdefault(node_id, self._topological_rank().get(node_id, 0))
        return depth

    def _ensure_acyclic_enabled_connections(self) -> None:
        """Disable edges that participate in cycles until enabled subgraph is acyclic."""
        max_iters = max(1, len(self.connection_genes))
        for _ in range(max_iters):
            node_ids = [int(node["id"]) for node in self.node_genes]
            indegree = {node_id: 0 for node_id in node_ids}
            outgoing: dict[int, list[dict[str, Any]]] = {node_id: [] for node_id in node_ids}
            enabled = [gene for gene in self.connection_genes if gene.get("enabled", True)]

            for gene in enabled:
                src = int(gene["in_node"])
                dst = int(gene["out_node"])
                outgoing.setdefault(src, []).append(gene)
                indegree[dst] = indegree.get(dst, 0) + 1

            queue = deque([node_id for node_id, deg in indegree.items() if deg == 0])
            seen = 0
            while queue:
                current = queue.popleft()
                seen += 1
                for edge in outgoing.get(current, []):
                    dst = int(edge["out_node"])
                    indegree[dst] -= 1
                    if indegree[dst] == 0:
                        queue.append(dst)

            if seen == len(node_ids):
                return

            cycle_edges = [
                gene
                for gene in enabled
                if indegree.get(int(gene["in_node"]), 0) > 0 and indegree.get(int(gene["out_node"]), 0) > 0
            ]
            if not cycle_edges:
                return

            edge_to_disable = max(cycle_edges, key=lambda gene: int(gene.get("innovation", -1)))
            edge_to_disable["enabled"] = False

    def _innovation_for(self, tracker: InnovationTracker | None, in_node: int, out_node: int) -> int:
        if tracker is not None:
            return tracker.get_connection_innovation(in_node, out_node)

        existing = [int(g.get("innovation", -1)) for g in self.connection_genes]
        return (max(existing) + 1) if existing else 0

    def _next_node_id(self) -> int:
        existing = [int(node["id"]) for node in self.node_genes]
        return (max(existing) + 1) if existing else 0

    def _nodes_by_type(self, node_type: str) -> list[dict[str, Any]]:
        return sorted(
            [node for node in self.node_genes if node.get("type") == node_type],
            key=lambda node: int(node["id"]),
        )

    def _node_type(self, node_id: int) -> str:
        for node in self.node_genes:
            if int(node["id"]) == int(node_id):
                return str(node.get("type", "hidden"))
        return "hidden"

    def _topological_rank(self) -> dict[int, int]:
        input_ids = [int(node["id"]) for node in self._nodes_by_type("input")]
        hidden_ids = [int(node["id"]) for node in self._nodes_by_type("hidden")]
        output_ids = [int(node["id"]) for node in self._nodes_by_type("output")]

        rank: dict[int, int] = {}
        for node_id in input_ids:
            rank[node_id] = 0
        for offset, node_id in enumerate(hidden_ids, start=1):
            rank[node_id] = offset
        base = len(hidden_ids) + 1
        for offset, node_id in enumerate(output_ids, start=base):
            rank[node_id] = offset
        return rank

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)
