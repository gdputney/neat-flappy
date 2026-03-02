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

        node_lookup = {int(node["id"]): node for node in self.node_genes}
        self._repair_cycles_in_enabled_graph()
        enabled_edges = [gene for gene in self.connection_genes if gene.get("enabled", True)]

        indegree = {int(node["id"]): 0 for node in self.node_genes}
        outgoing: dict[int, list[dict[str, Any]]] = {int(node["id"]): [] for node in self.node_genes}
        for edge in enabled_edges:
            in_node = int(edge["in_node"])
            out_node = int(edge["out_node"])
            indegree[out_node] = indegree.get(out_node, 0) + 1
            outgoing.setdefault(in_node, []).append(edge)

        queue = [node_id for node_id, degree in indegree.items() if degree == 0]
        topo_order: list[int] = []
        while queue:
            current = queue.pop(0)
            topo_order.append(current)
            for edge in outgoing.get(current, []):
                out_node = int(edge["out_node"])
                indegree[out_node] -= 1
                if indegree[out_node] == 0:
                    queue.append(out_node)

        if len(topo_order) != len(indegree):
            self._repair_cycles_in_enabled_graph()
            enabled_edges = [gene for gene in self.connection_genes if gene.get("enabled", True)]
            indegree = {int(node["id"]): 0 for node in self.node_genes}
            outgoing = {int(node["id"]): [] for node in self.node_genes}
            for edge in enabled_edges:
                in_node = int(edge["in_node"])
                out_node = int(edge["out_node"])
                indegree[out_node] = indegree.get(out_node, 0) + 1
                outgoing.setdefault(in_node, []).append(edge)

            queue = [node_id for node_id, degree in indegree.items() if degree == 0]
            topo_order = []
            while queue:
                current = queue.pop(0)
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
        if not self.would_create_cycle(in_node, out_node):
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

    def _add_connection_mutation(self, tracker: InnovationTracker | None = None) -> None:
        node_ids = [int(node["id"]) for node in self.node_genes]
        if len(node_ids) < 2:
            return

        existing_pairs = {(int(g["in_node"]), int(g["out_node"])) for g in self.connection_genes}
        topo_rank = self._topological_rank()

        candidates: list[tuple[int, int]] = []
        for a in node_ids:
            for b in node_ids:
                if a == b:
                    continue
                if (a, b) in existing_pairs:
                    continue
                if topo_rank.get(a, 0) >= topo_rank.get(b, 0):
                    continue
                if self._node_type(b) == "input":
                    continue
                if self.would_create_cycle(a, b):
                    continue
                candidates.append((a, b))

        if not candidates:
            return

        in_node, out_node = random.choice(candidates)
        self.connection_genes.append(
            {
                "in_node": in_node,
                "out_node": out_node,
                "weight": random.uniform(-1.0, 1.0),
                "enabled": True,
                "innovation": self._innovation_for(tracker, in_node, out_node),
            }
        )

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

        return Genome(node_genes=child_nodes, connection_genes=child_connections, fitness=0.0)

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

    def _has_enabled_path(self, src: int, dst: int) -> bool:
        src = int(src)
        dst = int(dst)
        if src == dst:
            return True

        adjacency: dict[int, list[int]] = {}
        for gene in self.connection_genes:
            if not gene.get("enabled", True):
                continue
            in_node = int(gene["in_node"])
            out_node = int(gene["out_node"])
            adjacency.setdefault(in_node, []).append(out_node)

        queue: deque[int] = deque([src])
        visited = {src}
        while queue:
            current = queue.popleft()
            for nxt in adjacency.get(current, []):
                if nxt == dst:
                    return True
                if nxt in visited:
                    continue
                visited.add(nxt)
                queue.append(nxt)
        return False

    def would_create_cycle(self, src: int, dst: int) -> bool:
        src = int(src)
        dst = int(dst)
        if src == dst:
            return True
        return self._has_enabled_path(dst, src)

    def _repair_cycles_in_enabled_graph(self) -> None:
        while True:
            cyclic_nodes = self._enabled_cyclic_nodes()
            if not cyclic_nodes:
                return

            candidates = [
                gene
                for gene in self.connection_genes
                if gene.get("enabled", True)
                and int(gene["in_node"]) in cyclic_nodes
                and int(gene["out_node"]) in cyclic_nodes
            ]
            if not candidates:
                return

            edge_to_disable = max(
                candidates,
                key=lambda g: (int(g.get("innovation", -1)), -abs(float(g.get("weight", 0.0)))),
            )
            edge_to_disable["enabled"] = False

    def _enabled_cyclic_nodes(self) -> set[int]:
        node_ids = {int(node["id"]) for node in self.node_genes}
        indegree = {node_id: 0 for node_id in node_ids}
        outgoing: dict[int, list[int]] = {node_id: [] for node_id in node_ids}

        for gene in self.connection_genes:
            if not gene.get("enabled", True):
                continue
            in_node = int(gene["in_node"])
            out_node = int(gene["out_node"])
            outgoing.setdefault(in_node, []).append(out_node)
            indegree[out_node] = indegree.get(out_node, 0) + 1

        queue: deque[int] = deque(node_id for node_id, degree in indegree.items() if degree == 0)
        processed: set[int] = set()
        while queue:
            current = queue.popleft()
            processed.add(current)
            for out_node in outgoing.get(current, []):
                indegree[out_node] -= 1
                if indegree[out_node] == 0:
                    queue.append(out_node)

        return node_ids - processed

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
