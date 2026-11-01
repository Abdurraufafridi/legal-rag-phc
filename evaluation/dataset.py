"""Load and validate the versioned retrieval evaluation dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def load_dataset(
    queries_path: Path,
    qrels_path: Path,
    document_folder: Path | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]], Dict[str, Any]]:
    queries = load_jsonl(queries_path)
    qrels_rows = load_jsonl(qrels_path)

    if not queries:
        raise ValueError("Evaluation query file is empty")
    if len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("Duplicate query_id in evaluation queries")

    qrels: Dict[str, Dict[str, int]] = {}
    for row in qrels_rows:
        qid = row["query_id"]
        doc_id = row["doc_id"]
        grade = int(row["grade"])
        if grade < 0:
            raise ValueError(f"Negative grade for {qid}/{doc_id}")
        qrels.setdefault(qid, {})[doc_id] = grade

    query_ids = {q["query_id"] for q in queries}
    unknown = set(qrels) - query_ids
    if unknown:
        raise ValueError(f"Qrels reference unknown query IDs: {sorted(unknown)}")

    if document_folder is not None:
        available = {p.name for p in Path(document_folder).iterdir() if p.is_file()}
        missing = sorted({doc for rel in qrels.values() for doc in rel} - available)
        if missing:
            raise ValueError(f"Qrels reference missing documents: {missing[:10]}")

    version = sorted({q.get("dataset_version", "unversioned") for q in queries})
    meta = {
        "dataset_version": ",".join(version),
        "n_queries": len(queries),
        "n_qrels": len(qrels_rows),
        "n_queries_with_qrels": sum(1 for qid in qrels if qrels[qid]),
        "categories": _category_counts(queries),
        "queries_sha256": _sha256(queries_path),
        "qrels_sha256": _sha256(qrels_path),
        "n_documents_labeled": len({d for rel in qrels.values() for d in rel}),
    }
    return queries, qrels, meta


def _category_counts(queries: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for query in queries:
        category = query.get("category", "unknown")
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()
