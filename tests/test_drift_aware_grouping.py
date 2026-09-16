import unittest

from hcsmoe.merging.drift_aware_grouping import (
    greedy_drift_aware_grouping,
    minmax_normalize,
)


class DriftAwareGroupingTest(unittest.TestCase):
    def test_greedy_exact_8_to_4_candidate_counts_and_deterministic_tie_breaking(self):
        result = greedy_drift_aware_grouping(
            [(index,) for index in range(8)],
            target_num_groups=4,
            evaluate_partition=lambda _partition: {"c_out": 1.0, "c_route": 1.0},
            lambda_route=0.5,
        )
        self.assertEqual(len(result["groups"]), 4)
        self.assertEqual(
            [len(step["candidates"]) for step in result["merge_trace"]],
            [28, 21, 15, 10],
        )
        self.assertEqual(result["merge_trace"][0]["selected_candidate"]["group_a"], [0])
        self.assertEqual(result["merge_trace"][0]["selected_candidate"]["group_b"], [1])


    def test_minmax_normalization_and_constant_metric(self):
        self.assertEqual(minmax_normalize([5.0, 5.0, 5.0]), [0.0, 0.0, 0.0])
        normalized = minmax_normalize([2.0, 4.0, 6.0])
        self.assertEqual(normalized[0], 0.0)
        self.assertAlmostEqual(normalized[1], 0.5)
        self.assertAlmostEqual(normalized[2], 1.0)


    def test_final_layer_route_unavailable_uses_output_cost_only(self):
        def evaluate(partition):
            merged = max(partition, key=len)
            return {"c_out": float(sum(merged)), "c_route": None}

        result = greedy_drift_aware_grouping(
            [(0,), (1,), (2,)],
            target_num_groups=2,
            evaluate_partition=evaluate,
            lambda_route=1.0,
        )
        selected = result["merge_trace"][0]["selected_candidate"]
        self.assertEqual(selected["resulting_group"], [0, 1])
        self.assertIsNone(selected["normalized_c_route"])
        self.assertEqual(selected["c_total"], selected["normalized_c_out"])


if __name__ == "__main__":
    unittest.main()
