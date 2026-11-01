import unittest
from collections import OrderedDict
from types import SimpleNamespace

from rag_core import (
    RetrievalConfig,
    aggregate_points,
    filter_documents_by_year,
    rank_documents,
)


def point(file_name, text, chunk_id, score):
    return SimpleNamespace(
        payload={"file_name": file_name, "text": text, "chunk_id": chunk_id},
        score=score,
    )


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = []

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        self.calls.append((len(pairs), batch_size))
        return self.scores


class AggregationTests(unittest.TestCase):
    def test_grouping_preserves_rank_and_best_score(self):
        points = [
            point("A.txt", "a one", "a1", 0.9),
            point("B.txt", "b one", "b1", 0.8),
            point("A.txt", "a two", "a2", 0.7),
        ]
        docs = aggregate_points(points)
        self.assertEqual(list(docs), ["A.txt", "B.txt"])
        self.assertEqual(len(docs["A.txt"]["chunks"]), 2)
        self.assertEqual(docs["A.txt"]["ranks"], [0, 2])
        self.assertAlmostEqual(docs["A.txt"]["best_retrieval_score"], 0.9)

    def test_empty_text_is_ignored(self):
        docs = aggregate_points([point("A.txt", "   ", "a1", 1.0)])
        self.assertEqual(len(docs), 0)

    def test_dedupe_by_chunk_id(self):
        points = [point("A.txt", "same", "a1", 0.9), point("A.txt", "same", "a1", 0.8)]
        docs = aggregate_points(points, dedupe_chunks=True)
        self.assertEqual(len(docs["A.txt"]["chunks"]), 1)

    def test_year_filter_keeps_undated_documents(self):
        docs = OrderedDict(
            [
                ("CrA 2011 one.txt", {"file_name": "CrA 2011 one.txt"}),
                ("CrA 2015 two.txt", {"file_name": "CrA 2015 two.txt"}),
                ("undated.txt", {"file_name": "undated.txt"}),
            ]
        )
        filtered = filter_documents_by_year(docs, 2010, 2012)
        self.assertEqual(list(filtered), ["CrA 2011 one.txt", "undated.txt"])


class RankingTests(unittest.TestCase):
    def _docs(self):
        points = [
            point("A.txt", "short", "a1", 0.99),
            point("A.txt", "much longer text", "a2", 0.98),
            point("B.txt", "medium", "b1", 0.97),
        ]
        return aggregate_points(points)

    def test_rerank_uses_max_logit_per_document(self):
        ce = FakeCrossEncoder([0.0, 2.0, 1.0])
        ranked = rank_documents("q", self._docs(), ce, rerank=True)
        self.assertEqual(ranked[0]["file_name"], "A.txt")
        self.assertGreater(ranked[0]["score"], ranked[1]["score"])
        self.assertEqual(ce.calls[0][0], 3)

    def test_chunk_cap_changes_ranking(self):
        ce = FakeCrossEncoder([0.0, 1.0])
        ranked = rank_documents(
            "q", self._docs(), ce, rerank=True, max_chunks_per_doc=1
        )
        self.assertEqual(ranked[0]["file_name"], "B.txt")
        self.assertEqual(ce.calls[0][0], 2)  # one chunk from A, one from B

    def test_no_rerank_uses_retrieval_score(self):
        ranked = rank_documents("q", self._docs(), None, rerank=False)
        self.assertEqual(ranked[0]["file_name"], "A.txt")
        self.assertFalse(ranked[0]["reranked"])

    def test_rerank_max_docs(self):
        ce = FakeCrossEncoder([2.0, 0.0, 1.0])
        ranked = rank_documents("q", self._docs(), ce, rerank=True, rerank_max_docs=2)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ce.calls[0][0], 3)

    def test_empty_documents(self):
        self.assertEqual(rank_documents("q", OrderedDict(), None), [])


class ConfigTests(unittest.TestCase):
    def test_round_trip(self):
        cfg = RetrievalConfig(name="x", prefetch_limit=20)
        self.assertEqual(RetrievalConfig.from_dict(cfg.to_dict()), cfg)

    def test_unknown_key_rejected(self):
        with self.assertRaises(ValueError):
            RetrievalConfig.from_dict({"name": "x", "unknown_option": 1})


if __name__ == "__main__":
    unittest.main()
