import unittest

from neat_core import Genome, InnovationTracker


class NeatCycleSafetyTests(unittest.TestCase):
    def _base_genome(self) -> Genome:
        nodes = [
            {"id": 0, "type": "input", "bias": 0.0},
            {"id": 1, "type": "input", "bias": 0.0},
            {"id": 2, "type": "output", "bias": 0.0},
        ]
        conns = [
            {"in_node": 0, "out_node": 2, "weight": 0.5, "enabled": True, "innovation": 0},
            {"in_node": 1, "out_node": 2, "weight": -0.5, "enabled": True, "innovation": 1},
        ]
        return Genome(node_genes=nodes, connection_genes=conns)

    def test_add_connection_mutation_does_not_introduce_cycle(self) -> None:
        tracker = InnovationTracker(next_innovation=2)
        genome = self._base_genome()

        # Force a hidden node path 0 -> 3 -> 2.
        genome.node_genes.append({"id": 3, "type": "hidden", "bias": 0.0})
        genome.connection_genes.append(
            {"in_node": 0, "out_node": 3, "weight": 1.0, "enabled": True, "innovation": 2}
        )
        genome.connection_genes.append(
            {"in_node": 3, "out_node": 2, "weight": 1.0, "enabled": True, "innovation": 3}
        )

        for _ in range(200):
            genome._add_connection_mutation(tracker)
            # No enabled edge may close a directed cycle.
            for gene in genome.connection_genes:
                if not gene.get("enabled", True):
                    continue
                self.assertFalse(
                    genome._has_enabled_path(int(gene["out_node"]), int(gene["in_node"])),
                    msg=f"Found cycle-closing edge: {gene}",
                )

    def test_activate_repairs_cyclic_enabled_graph(self) -> None:
        genome = self._base_genome()
        # Inject explicit cycle between output and hidden.
        genome.node_genes.append({"id": 3, "type": "hidden", "bias": 0.0})
        genome.connection_genes.extend(
            [
                {"in_node": 0, "out_node": 3, "weight": 1.0, "enabled": True, "innovation": 2},
                {"in_node": 3, "out_node": 2, "weight": 1.0, "enabled": True, "innovation": 3},
                {"in_node": 2, "out_node": 3, "weight": 1.0, "enabled": True, "innovation": 4},
            ]
        )

        output = genome.activate([0.2, 0.8])
        self.assertEqual(len(output), 1)

    def test_toggle_enable_does_not_create_cycle(self) -> None:
        genome = self._base_genome()
        genome.node_genes.append({"id": 3, "type": "hidden", "bias": 0.0})
        genome.connection_genes.extend(
            [
                {"in_node": 0, "out_node": 3, "weight": 1.0, "enabled": True, "innovation": 2},
                {"in_node": 3, "out_node": 2, "weight": 1.0, "enabled": True, "innovation": 3},
                {"in_node": 2, "out_node": 3, "weight": 0.7, "enabled": False, "innovation": 4},
            ]
        )

        for _ in range(80):
            genome._toggle_connection_enabled()
            for gene in genome.connection_genes:
                if not gene.get("enabled", True):
                    continue
                self.assertFalse(
                    genome._has_enabled_path(int(gene["out_node"]), int(gene["in_node"])),
                    msg=f"Toggle produced cycle-closing edge: {gene}",
                )

    def test_mutate_never_produces_cyclic_enabled_graph(self) -> None:
        tracker = InnovationTracker(next_innovation=5)
        genome = self._base_genome()
        genome.node_genes.append({"id": 3, "type": "hidden", "bias": 0.0})
        genome.connection_genes.extend(
            [
                {"in_node": 0, "out_node": 3, "weight": 1.0, "enabled": True, "innovation": 2},
                {"in_node": 3, "out_node": 2, "weight": 1.0, "enabled": True, "innovation": 3},
                {"in_node": 2, "out_node": 3, "weight": 1.0, "enabled": False, "innovation": 4},
            ]
        )

        for _ in range(300):
            genome.mutate(tracker)
            for gene in genome.connection_genes:
                if not gene.get("enabled", True):
                    continue
                self.assertFalse(
                    genome._has_enabled_path(int(gene["out_node"]), int(gene["in_node"])),
                    msg=f"Mutate produced cycle-closing edge: {gene}",
                )


if __name__ == "__main__":
    unittest.main()
