import unittest
from pathlib import Path

from evaluation.dataset import load_dataset

ROOT = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.queries, cls.qrels, cls.meta = load_dataset(
            ROOT / "evaluation" / "queries" / "evaluation_queries.jsonl",
            ROOT / "evaluation" / "queries" / "relevance_judgments.jsonl",
            document_folder=ROOT / "FYP1_extracted_text",
        )

    def test_counts(self):
        self.assertEqual(len(self.queries), 44)
        self.assertEqual(self.meta["n_queries"], 44)
        self.assertEqual(self.meta["n_qrels"], 107)
        self.assertEqual(self.meta["n_queries_with_qrels"], 40)
        self.assertEqual(self.meta["n_documents_labeled"], 74)

    def test_versions_and_hashes(self):
        self.assertEqual(self.meta["dataset_version"], "v1.0.0")
        self.assertEqual(len(self.meta["queries_sha256"]), 64)
        self.assertEqual(len(self.meta["qrels_sha256"]), 64)

    def test_out_of_domain_queries_have_no_qrels(self):
        ood = [q["query_id"] for q in self.queries if q["category"] == "out_of_domain"]
        self.assertEqual(len(ood), 4)
        for qid in ood:
            self.assertEqual(self.qrels.get(qid, {}), {})

    def test_all_in_domain_have_qrels(self):
        for query in self.queries:
            if query["category"] != "out_of_domain":
                self.assertTrue(self.qrels.get(query["query_id"]), query["query_id"])

    def test_grades_are_valid(self):
        for qid, rels in self.qrels.items():
            for doc, grade in rels.items():
                self.assertIn(grade, (1, 2, 3), f"{qid}/{doc}")


if __name__ == "__main__":
    unittest.main()
