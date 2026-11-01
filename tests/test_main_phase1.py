import asyncio
import unittest
from types import SimpleNamespace

import numpy as np

import main


class FakeEmbeddingModel:
    def encode(self, text, normalize_embeddings=True, show_progress_bar=False, batch_size=32):
        if isinstance(text, list):
            return np.tile(np.array([0.1, 0.2, 0.3]), (len(text), 1))
        return np.array([0.1, 0.2, 0.3])


class FakeSparseRow:
    indices = np.array([0, 2])
    data = np.array([0.5, 0.5])


class FakeSparseVectorizer:
    def transform(self, texts):
        return [FakeSparseRow() for _ in texts]


class FakeQdrantClient:
    def __init__(self, points):
        self.points = points
        self.last_limit = None
        self.last_filter = None

    def query_points(self, **kwargs):
        self.last_limit = kwargs["limit"]
        self.last_filter = kwargs.get("query_filter")
        points = self.points
        if self.last_filter is not None:
            points = [p for p in points if p.payload.get("level") == "chunk"]
        return SimpleNamespace(points=points[: kwargs["limit"]])

    def upsert(self, collection_name, points):
        self.upserted = points


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = []

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        self.calls.append((len(pairs), batch_size, show_progress_bar))
        return np.asarray(self.scores, dtype=float)


def point(file_name, level, text="text", uploaded_by=None):
    payload = {"file_name": file_name, "level": level, "text": text}
    if uploaded_by:
        payload["uploaded_by"] = uploaded_by
    return SimpleNamespace(payload=payload)


class RetrievalCandidateTests(unittest.TestCase):
    def setUp(self):
        self.originals = (
            main.embedding_model,
            main.sparse_vectorizer,
            main.qdrant_client,
            main.cross_encoder,
        )
        main.embedding_model = FakeEmbeddingModel()
        main.sparse_vectorizer = FakeSparseVectorizer()

    def tearDown(self):
        (
            main.embedding_model,
            main.sparse_vectorizer,
            main.qdrant_client,
            main.cross_encoder,
        ) = self.originals

    def test_chunk_filter_keeps_only_chunk_level_points(self):
        query_filter = main._chunk_level_filter(include_document_level=False)
        self.assertIsNotNone(query_filter)
        self.assertEqual(query_filter.must[0].key, "level")
        self.assertEqual(query_filter.must[0].match.value, "chunk")
        self.assertIsNone(main._chunk_level_filter(include_document_level=True))

    def test_hybrid_search_requests_chunk_only_candidates(self):
        points = [
            point("corpus.txt", "document"),
            point("chunk-a.txt", "chunk"),
            point("chunk-b.txt", "chunk"),
        ]
        fake_qdrant = FakeQdrantClient(points)
        main.qdrant_client = fake_qdrant

        results = main.hybrid_search(
            "kidnapping", limit=2, include_document_level=False
        )

        self.assertEqual(
            [p.payload["file_name"] for p in results],
            ["chunk-a.txt", "chunk-b.txt"],
        )
        self.assertEqual(fake_qdrant.last_limit, 2)
        self.assertIsNotNone(fake_qdrant.last_filter)

    def test_hybrid_search_can_include_document_level(self):
        points = [
            point("corpus.txt", "document"),
            point("chunk-a.txt", "chunk"),
            point("chunk-b.txt", "chunk"),
        ]
        fake_qdrant = FakeQdrantClient(points)
        main.qdrant_client = fake_qdrant

        results = main.hybrid_search(
            "kidnapping", limit=2, include_document_level=True
        )

        self.assertEqual(
            [p.payload["file_name"] for p in results],
            ["corpus.txt", "chunk-a.txt"],
        )
        self.assertEqual(fake_qdrant.last_limit, 2)
        self.assertIsNone(fake_qdrant.last_filter)


class RerankTests(unittest.TestCase):
    def setUp(self):
        self.original = main.cross_encoder

    def tearDown(self):
        main.cross_encoder = self.original

    def test_all_chunks_are_scored_in_one_batch(self):
        main.cross_encoder = FakeCrossEncoder([0.0, 2.0, 1.0])
        documents = {
            "A.txt": [{"text": "a one"}, {"text": "a two"}],
            "B.txt": [{"text": "b one"}],
            "C.txt": [{"text": ""}],
        }

        ranked = main.rerank_documents("query", documents)

        self.assertEqual([name for name, _ in ranked], ["A.txt", "B.txt"])
        self.assertEqual(main.cross_encoder.calls, [(3, main.RERANK_BATCH_SIZE, False)])

    def test_empty_documents_return_empty(self):
        main.cross_encoder = FakeCrossEncoder([])
        self.assertEqual(main.rerank_documents("query", {}), [])
        self.assertEqual(main.cross_encoder.calls, [])


class SummaryCacheTests(unittest.TestCase):
    def setUp(self):
        self.original_size = main.SUMMARY_CACHE_SIZE
        main.SUMMARY_CACHE_SIZE = 2
        with main._summary_cache_lock:
            main._summary_cache.clear()

    def tearDown(self):
        main.SUMMARY_CACHE_SIZE = self.original_size
        with main._summary_cache_lock:
            main._summary_cache.clear()

    def test_lru_eviction_and_hit(self):
        main.cache_summary("a", "summary-a")
        main.cache_summary("b", "summary-b")
        self.assertEqual(main.get_cached_summary("a"), "summary-a")

        main.cache_summary("c", "summary-c")

        self.assertEqual(main.get_cached_summary("a"), "summary-a")
        self.assertIsNone(main.get_cached_summary("b"))
        self.assertEqual(main.get_cached_summary("c"), "summary-c")

    def test_cache_disabled(self):
        main.SUMMARY_CACHE_SIZE = 0
        main.cache_summary("a", "summary-a")
        self.assertIsNone(main.get_cached_summary("a"))


if __name__ == "__main__":
    unittest.main()


class UploadIndexTests(unittest.TestCase):
    def setUp(self):
        self.originals = (main.embedding_model, main.sparse_vectorizer, main.qdrant_client)
        main.embedding_model = FakeEmbeddingModel()
        main.sparse_vectorizer = FakeSparseVectorizer()
        self.qdrant = FakeQdrantClient([])
        main.qdrant_client = self.qdrant

    def tearDown(self):
        main.embedding_model, main.sparse_vectorizer, main.qdrant_client = self.originals

    def test_upload_is_indexed_as_chunk_level_points(self):
        text = " ".join(f"word{i}" for i in range(400))
        count = main.index_uploaded_document(text, "upload.txt", "user@example.com")

        self.assertGreater(count, 1)
        self.assertEqual(len(self.qdrant.upserted), count)
        for point in self.qdrant.upserted:
            self.assertEqual(point.payload["level"], "chunk")
            self.assertEqual(point.payload["uploaded_by"], "user@example.com")
            self.assertEqual(point.payload["file_name"], "upload.txt")


class SummaryEndpointTests(unittest.TestCase):
    def setUp(self):
        self.original_call_llm = main.call_llm
        self.original_size = main.SUMMARY_CACHE_SIZE
        main.SUMMARY_CACHE_SIZE = 8
        with main._summary_cache_lock:
            main._summary_cache.clear()
        self.calls = []

    def tearDown(self):
        main.call_llm = self.original_call_llm
        main.SUMMARY_CACHE_SIZE = self.original_size
        with main._summary_cache_lock:
            main._summary_cache.clear()

    def test_second_summary_request_uses_cache(self):
        def fake_call_llm(messages, max_tokens=512, temperature=0.2):
            self.calls.append(messages)
            return "cached summary"

        main.call_llm = fake_call_llm
        request = {"text": "A judgment about bail."}

        first = asyncio.run(main.summarize_judgment(request, "user@example.com"))
        second = asyncio.run(main.summarize_judgment(request, "user@example.com"))

        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(second["summary"], "cached summary")
        self.assertEqual(len(self.calls), 1)


class UploadChunkingTests(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(main.chunk_document_text("short text"), ["short text"])

    def test_empty_text_has_no_chunks(self):
        self.assertEqual(main.chunk_document_text("   "), [])

    def test_long_text_is_split_with_overlap(self):
        text = " ".join(f"word{i}" for i in range(400))
        chunks = main.chunk_document_text(text, max_chars=200, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 200 for chunk in chunks))
        self.assertEqual(" ".join(chunks).split()[0], "word0")
