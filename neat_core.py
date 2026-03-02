"""Core NEAT genome abstractions for Flappy Bird experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any


@dataclass
class Genome:
    """Represents an evolvable feedforward NEAT-style genome.

    Attributes:
        node_genes: Optional metadata describing node layout.
        connection_genes: Optional metadata describing connection layout.
        fitness: Scalar genome fitness value.
        weight_layers: Feedforward weight matrices (layer by layer).
        bias_layers: Feedforward bias vectors (layer by layer).
    """

    node_genes: list[Any] = field(default_factory=list)
    connection_genes: list[Any] = field(default_factory=list)
    fitness: float = 0.0
    weight_layers: list[list[list[float]]] = field(default_factory=list)
    bias_layers: list[list[float]] = field(default_factory=list)

    def activate(self, inputs: list[float]) -> list[float]:
        """Run a feedforward pass with sigmoid activation."""
        if not self.weight_layers:
            raise ValueError("Genome has no weight layers; cannot activate network")
        if len(self.weight_layers) != len(self.bias_layers):
            raise ValueError("weight_layers and bias_layers must have the same number of layers")

        activations = [float(value) for value in inputs]

        for layer_index, (layer_weights, layer_biases) in enumerate(
            zip(self.weight_layers, self.bias_layers)
        ):
            if len(layer_weights) != len(layer_biases):
                raise ValueError(
                    f"Layer {layer_index} has mismatched neuron counts between weights and biases"
                )

            next_activations: list[float] = []
            for neuron_index, (neuron_weights, neuron_bias) in enumerate(
                zip(layer_weights, layer_biases)
            ):
                if len(neuron_weights) != len(activations):
                    raise ValueError(
                        "Layer "
                        f"{layer_index} neuron {neuron_index} expects {len(neuron_weights)} inputs "
                        f"but received {len(activations)}"
                    )

                weighted_sum = sum(
                    weight * value for weight, value in zip(neuron_weights, activations)
                ) + neuron_bias
                next_activations.append(self._sigmoid(weighted_sum))

            activations = next_activations

        return activations

    def mutate(self) -> None:
        """Apply core NEAT-style mutations to this genome.

        Includes:
        - perturbing existing connection weights,
        - adding a new node by splitting an existing connection,
        - adding a new connection between existing nodes.
        """
        self._perturb_connection_weights()

        if random.random() < 0.15:
            self._add_node_mutation()

        if random.random() < 0.25:
            self._add_connection_mutation()

    def _perturb_connection_weights(self, sigma: float = 0.3, reset_chance: float = 0.1) -> None:
        """Randomly perturb connection weights in genes and layer matrices."""
        for connection in self.connection_genes:
            if not isinstance(connection, dict):
                continue
            if "weight" not in connection:
                continue

            if random.random() < reset_chance:
                connection["weight"] = random.uniform(-1.0, 1.0)
            else:
                connection["weight"] = float(connection["weight"]) + random.gauss(0.0, sigma)

        for layer in self.weight_layers:
            for neuron_weights in layer:
                for index, weight in enumerate(neuron_weights):
                    if random.random() < reset_chance:
                        neuron_weights[index] = random.uniform(-1.0, 1.0)
                    else:
                        neuron_weights[index] = float(weight) + random.gauss(0.0, sigma)

    def _add_node_mutation(self) -> None:
        """Split one enabled connection into two connections through a new node."""
        enabled_connections = [
            gene
            for gene in self.connection_genes
            if isinstance(gene, dict)
            and gene.get("enabled", True)
            and "in_node" in gene
            and "out_node" in gene
            and "weight" in gene
        ]

        if not enabled_connections:
            return

        connection = random.choice(enabled_connections)
        connection["enabled"] = False

        new_node_id = self._next_node_id()
        self.node_genes.append(new_node_id)

        in_node = int(connection["in_node"])
        out_node = int(connection["out_node"])
        original_weight = float(connection["weight"])

        self.connection_genes.append(
            {
                "in_node": in_node,
                "out_node": new_node_id,
                "weight": 1.0,
                "enabled": True,
            }
        )
        self.connection_genes.append(
            {
                "in_node": new_node_id,
                "out_node": out_node,
                "weight": original_weight,
                "enabled": True,
            }
        )

    def _add_connection_mutation(self) -> None:
        """Add a new connection gene between two distinct existing nodes."""
        node_ids = self._node_ids()
        if len(node_ids) < 2:
            if not node_ids:
                self.node_genes.extend([0, 1])
            elif len(node_ids) == 1:
                self.node_genes.append(self._next_node_id())
            node_ids = self._node_ids()

        existing_pairs = {
            (int(gene["in_node"]), int(gene["out_node"]))
            for gene in self.connection_genes
            if isinstance(gene, dict) and "in_node" in gene and "out_node" in gene
        }

        candidates = [
            (a, b)
            for a in node_ids
            for b in node_ids
            if a != b and (a, b) not in existing_pairs
        ]
        if not candidates:
            return

        in_node, out_node = random.choice(candidates)
        self.connection_genes.append(
            {
                "in_node": in_node,
                "out_node": out_node,
                "weight": random.uniform(-1.0, 1.0),
                "enabled": True,
            }
        )

    def _node_ids(self) -> list[int]:
        """Return node IDs as integers from heterogeneous node gene formats."""
        ids: list[int] = []
        for node in self.node_genes:
            if isinstance(node, int):
                ids.append(node)
            elif isinstance(node, dict) and "id" in node:
                ids.append(int(node["id"]))
        return sorted(set(ids))

    def _next_node_id(self) -> int:
        """Get the next available node id."""
        existing = self._node_ids()
        if not existing:
            return 0
        return max(existing) + 1

    def crossover(self, other: "Genome") -> "Genome":
        """Combine two parent genomes into a child genome.

        Matching connection genes are identified by innovation number when
        present, otherwise by (in_node, out_node) pair. For matching genes,
        the child inherits one parent's version at random. Disjoint/excess
        genes are inherited from the fitter parent (or from either parent when
        fitness ties).
        """
        if not isinstance(other, Genome):
            raise TypeError("Genome.crossover() requires another Genome")

        fitter_parent, other_parent = self, other
        if other.fitness > self.fitness:
            fitter_parent, other_parent = other, self

        inherit_from_both = self.fitness == other.fitness

        self_genes = {
            self._connection_key(gene): gene
            for gene in self.connection_genes
            if isinstance(gene, dict)
        }
        other_genes = {
            self._connection_key(gene): gene
            for gene in other.connection_genes
            if isinstance(gene, dict)
        }

        all_keys = set(self_genes) | set(other_genes)
        child_connections: list[Any] = []
        for key in sorted(all_keys, key=str):
            gene_a = self_genes.get(key)
            gene_b = other_genes.get(key)

            chosen_gene: dict[str, Any] | None = None
            if gene_a is not None and gene_b is not None:
                chosen_gene = random.choice([gene_a, gene_b]).copy()
            elif inherit_from_both:
                chosen_gene = (gene_a or gene_b).copy() if (gene_a or gene_b) else None
            else:
                fitter_gene = (
                    fitter_parent.connection_genes and self._gene_from_parent(fitter_parent, key)
                )
                if fitter_gene is not None:
                    chosen_gene = fitter_gene.copy()

            if chosen_gene is not None:
                child_connections.append(chosen_gene)

        node_union = self._node_id_set() | other._node_id_set()
        child_nodes: list[Any] = sorted(node_union)

        child_weight_layers = self._blend_layers(self.weight_layers, other.weight_layers)
        child_bias_layers = self._blend_layers(self.bias_layers, other.bias_layers)

        child = Genome(
            node_genes=child_nodes,
            connection_genes=child_connections,
            weight_layers=child_weight_layers,
            bias_layers=child_bias_layers,
            fitness=0.0,
        )
        return child

    def _connection_key(self, gene: dict[str, Any]) -> tuple[str, int, int] | tuple[str, int]:
        """Build a stable key for matching genes across genomes."""
        if "innovation" in gene:
            return ("innovation", int(gene["innovation"]))

        in_node = int(gene.get("in_node", -1))
        out_node = int(gene.get("out_node", -1))
        return ("pair", in_node, out_node)

    def _gene_from_parent(
        self,
        parent: "Genome",
        key: tuple[str, int, int] | tuple[str, int],
    ) -> dict[str, Any] | None:
        """Fetch a parent connection gene by match key."""
        for gene in parent.connection_genes:
            if not isinstance(gene, dict):
                continue
            if self._connection_key(gene) == key:
                return gene
        return None

    def _node_id_set(self) -> set[int]:
        """Return node IDs as a set for crossover union operations."""
        return set(self._node_ids())

    @staticmethod
    def _blend_layers(
        first: list[list[Any]],
        second: list[list[Any]],
    ) -> list[list[Any]]:
        """Blend two layered parameter structures by random per-neuron inheritance."""
        if not first:
            return [[item for item in row] for row in second]
        if not second:
            return [[item for item in row] for row in first]

        max_layers = max(len(first), len(second))
        blended: list[list[Any]] = []

        for layer_idx in range(max_layers):
            layer_a = first[layer_idx] if layer_idx < len(first) else []
            layer_b = second[layer_idx] if layer_idx < len(second) else []
            max_neurons = max(len(layer_a), len(layer_b))
            layer_result: list[Any] = []

            for neuron_idx in range(max_neurons):
                has_a = neuron_idx < len(layer_a)
                has_b = neuron_idx < len(layer_b)
                if has_a and has_b:
                    source = random.choice([layer_a[neuron_idx], layer_b[neuron_idx]])
                elif has_a:
                    source = layer_a[neuron_idx]
                elif has_b:
                    source = layer_b[neuron_idx]
                else:
                    continue

                if isinstance(source, list):
                    layer_result.append(list(source))
                else:
                    layer_result.append(source)

            blended.append(layer_result)

        return blended

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)
