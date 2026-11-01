import math
import unittest

from evaluation.metrics import evaluate_rankings, macro_average, query_metrics


class QueryMetricTests(unittest.TestCase):
    def test_known_example(self):
        qrels = {"A": 3, "B": 2, "C": 1}
        ranking = ["A", "C", "X", "B"]
        m = query_metrics(ranking, qrels, cutoffs=(1, 3, 5, 10), mrr_cutoff=10)

        self.assertTrue(m["has_qrels"])
        self.assertEqual(m["relevant_count"], 3)
        self.assertAlmostEqual(m["recall@1"], 1 / 3)
        self.assertAlmostEqual(m["recall@3"], 2 / 3)
        self.assertAlmostEqual(m["precision@1"], 1.0)
        self.assertAlmostEqual(m["precision@3"], 2 / 3)
        self.assertAlmostEqual(m["mrr@10"], 1.0)
        self.assertEqual(m["first_relevant_rank"], 1)
        self.assertAlmostEqual(m["map@10"], (1.0 + 1.0 + 3 / 4) / 3)

        dcg = 7.0 + 1.0 / math.log2(3)
        idcg = 7.0 + 3.0 / math.log2(3) + 1.0 / math.log2(4)
        self.assertAlmostEqual(m["ndcg@3"], dcg / idcg)

    def test_mrr_uses_first_relevant(self):
        qrels = {"A": 1, "B": 1}
        m = query_metrics(["X", "B", "A"], qrels, cutoffs=(3,), mrr_cutoff=10)
        self.assertAlmostEqual(m["mrr@10"], 0.5)
        self.assertAlmostEqual(m["precision@3"], 2 / 3)
        self.assertAlmostEqual(m["recall@3"], 1.0)
        self.assertAlmostEqual(m["map@10"], (1 / 2 + 2 / 3) / 2)

    def test_empty_qrels_returns_none(self):
        m = query_metrics(["A", "B"], {}, cutoffs=(1, 5))
        self.assertFalse(m["has_qrels"])
        self.assertIsNone(m["recall@5"])
        self.assertIsNone(m["mrr@10"])
        self.assertIsNone(m["ndcg@5"])

    def test_no_results(self):
        m = query_metrics([], {"A": 3}, cutoffs=(5,))
        self.assertEqual(m["recall@5"], 0.0)
        self.assertEqual(m["precision@5"], 0.0)
        self.assertEqual(m["mrr@10"], 0.0)
        self.assertEqual(m["map@10"], 0.0)
        self.assertIsNone(m["first_relevant_rank"])

    def test_precision_denominator_is_k(self):
        m = query_metrics(["A"], {"A": 1, "B": 1}, cutoffs=(5,))
        self.assertAlmostEqual(m["precision@5"], 1 / 5)
        self.assertAlmostEqual(m["recall@5"], 0.5)

    def test_macro_average_ignores_none(self):
        per_query = [
            {"recall@5": 1.0, "has_qrels": True},
            {"recall@5": 0.0, "has_qrels": True},
            {"recall@5": None, "has_qrels": False},
        ]
        avg = macro_average(per_query, keys=["recall@5"])
        self.assertAlmostEqual(avg["recall@5"]["mean"], 0.5)
        self.assertEqual(avg["recall@5"]["n_queries"], 2)

    def test_evaluate_rankings_separates_ood(self):
        rankings = {"Q1": ["A"], "Q2": ["B"]}
        qrels = {"Q1": {"A": 3}, "Q2": {}}
        result = evaluate_rankings(rankings, qrels, cutoffs=(1,))
        self.assertEqual(result["n_queries_with_qrels"], 1)
        self.assertEqual(result["n_queries_without_qrels"], 1)
        self.assertAlmostEqual(result["metrics"]["recall@1"]["mean"], 1.0)
        self.assertIsNone(result["per_query"]["Q2"]["recall@1"])


if __name__ == "__main__":
    unittest.main()
