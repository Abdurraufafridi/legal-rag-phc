"""Document-level retrieval metrics.

Conventions (used consistently in every experiment):

* Rankings are lists of judgment `file_name` values, best first.
* A document is relevant if its qrel grade > 0.
* Recall@k = hits in top k / total known relevant documents.
* Precision@k = hits in top k / k. The denominator is always k, including
  when the system returned fewer than k results; empty results therefore
  receive 0 precision. This is the standard IR convention.
* MRR@10 = 1 / rank of the first relevant document in the top 10, else 0.
* AP@k = sum(precision at each relevant rank) / min(total relevant, k).
* nDCG@k uses gain = 2**grade - 1 and discount = log2(rank + 1);
  IDCG uses the highest-graded known relevant documents up to k.
* Queries with no known relevant judgments return None for all metrics and
  are excluded from macro averages. Their empty-result behavior is reported
  separately as an out-of-domain check.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _dcg(grades: Sequence[float]) -> float:
    return sum((2.0 ** g - 1.0) / math.log2(i + 2.0) for i, g in enumerate(grades))


def query_metrics(
    ranked_doc_ids: Sequence[str],
    qrels: Mapping[str, int],
    cutoffs: Sequence[int] = (1, 3, 5, 10),
    mrr_cutoff: int = 10,
) -> Dict[str, Optional[float]]:
    """Compute metrics for one query. Returns None when qrels are empty."""
    if not qrels:
        return {
            "has_qrels": False,
            **{f"recall@{k}": None for k in cutoffs},
            **{f"precision@{k}": None for k in cutoffs},
            **{f"ndcg@{k}": None for k in cutoffs},
            f"mrr@{mrr_cutoff}": None,
            f"map@{mrr_cutoff}": None,
            "first_relevant_rank": None,
            "relevant_count": 0,
        }

    relevant = {doc for doc, grade in qrels.items() if grade and grade > 0}
    total_relevant = len(relevant)
    grades = {doc: int(grade) for doc, grade in qrels.items() if grade and grade > 0}
    result: Dict[str, Optional[float]] = {
        "has_qrels": True,
        "relevant_count": total_relevant,
    }

    for k in cutoffs:
        top = list(ranked_doc_ids[:k])
        hits = sum(1 for doc in top if doc in relevant)
        result[f"recall@{k}"] = hits / total_relevant if total_relevant else None
        result[f"precision@{k}"] = hits / k if k else None

        dcg = _dcg([grades.get(doc, 0) for doc in top])
        ideal_grades = sorted(grades.values(), reverse=True)[:k]
        idcg = _dcg(ideal_grades)
        result[f"ndcg@{k}"] = (dcg / idcg) if idcg > 0 else None

    first_rank = None
    for idx, doc in enumerate(ranked_doc_ids[:mrr_cutoff], start=1):
        if doc in relevant:
            first_rank = idx
            break
    result[f"mrr@{mrr_cutoff}"] = (1.0 / first_rank) if first_rank else 0.0
    result["first_relevant_rank"] = first_rank

    # Average precision at the MRR cutoff.
    hits = 0
    precision_sum = 0.0
    for idx, doc in enumerate(ranked_doc_ids[:mrr_cutoff], start=1):
        if doc in relevant:
            hits += 1
            precision_sum += hits / idx
    denominator = min(total_relevant, mrr_cutoff)
    result[f"map@{mrr_cutoff}"] = precision_sum / denominator if denominator else None

    return result


def macro_average(
    per_query: Iterable[Mapping[str, Optional[float]]],
    keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Average only non-None values and report the number of contributing queries."""
    metrics = list(per_query)
    if keys is None:
        keys = [
            key
            for key in metrics[0].keys()
            if key
            not in ("has_qrels", "first_relevant_rank", "relevant_count")
        ] if metrics else []

    averages: Dict[str, Any] = {}
    for key in keys:
        values = [float(m[key]) for m in metrics if m.get(key) is not None]
        averages[key] = {
            "mean": (sum(values) / len(values)) if values else None,
            "n_queries": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return averages


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    qrels_by_query: Mapping[str, Mapping[str, int]],
    cutoffs: Sequence[int] = (1, 3, 5, 10),
    mrr_cutoff: int = 10,
) -> Dict[str, Any]:
    """Evaluate a query_id -> ranked doc list mapping."""
    per_query: Dict[str, Dict[str, Any]] = {}
    for query_id, qrels in qrels_by_query.items():
        per_query[query_id] = query_metrics(
            rankings.get(query_id, []), qrels, cutoffs=cutoffs, mrr_cutoff=mrr_cutoff
        )

    with_qrels = {qid: m for qid, m in per_query.items() if m.get("has_qrels")}
    without_qrels = {qid: m for qid, m in per_query.items() if not m.get("has_qrels")}

    return {
        "metrics": macro_average(with_qrels.values()),
        "per_query": per_query,
        "n_queries_with_qrels": len(with_qrels),
        "n_queries_without_qrels": len(without_qrels),
    }
