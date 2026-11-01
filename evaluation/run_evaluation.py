#!/usr/bin/env python
"""Run a reproducible retrieval experiment and store its numerical results.

Example:
    ./bin/python -m evaluation.run_evaluation \
        --config evaluation/configs/baseline_main.json \
        --output evaluation/results/baseline_results.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as app_config  # noqa: E402
from evaluation.dataset import load_dataset  # noqa: E402
from evaluation.metrics import evaluate_rankings  # noqa: E402
from rag_core import LegalRetriever, RetrievalConfig  # noqa: E402

QUERIES_PATH = ROOT / "evaluation" / "queries" / "evaluation_queries.jsonl"
QRELS_PATH = ROOT / "evaluation" / "queries" / "relevance_judgments.jsonl"
RESULTS_DIR = ROOT / "evaluation" / "results"
COMPARISON_CSV = RESULTS_DIR / "experiment_comparison.csv"


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(p / 100.0 * len(ordered)) - 1))
    return ordered[idx]


def latency_summary(rows: List[Dict[str, float]], keys: List[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"n_queries": len(rows)}
    for key in keys:
        values = [float(r[key]) for r in rows if r.get(key) is not None]
        if not values:
            summary[key] = None
            continue
        summary[key] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p95": percentile(values, 95),
            "min": min(values),
            "max": max(values),
        }
    return summary


def strip_ranking(ranked: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    out = []
    for idx, item in enumerate(ranked[:limit], start=1):
        out.append(
            {
                "rank": idx,
                "file_name": item["file_name"],
                "score": round(float(item["score"]), 8),
                "reranked": bool(item.get("reranked")),
            }
        )
    return out


def collection_metadata(retriever: LegalRetriever) -> Dict[str, Any]:
    try:
        info = retriever.client.get_collection(retriever.collection_name)
        vectors = info.config.params.vectors
        dense = vectors.get("dense") if isinstance(vectors, dict) else vectors
        return {
            "collection": retriever.collection_name,
            "points_count": info.points_count,
            "dense_size": getattr(dense, "size", None),
            "dense_distance": str(getattr(dense, "distance", None)),
            "sparse_vectors": list(info.config.params.sparse_vectors or {}),
        }
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"collection": retriever.collection_name, "error": str(exc)}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    config_data = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = RetrievalConfig.from_dict(config_data)
    if args.name:
        config.name = args.name

    queries, qrels, dataset_meta = load_dataset(
        Path(args.queries), Path(args.qrels), document_folder=app_config.TEXT_DIRECTORY
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = args.experiment_id or f"{config.name}_{timestamp}"
    output_path = Path(args.output).resolve() if args.output else (RESULTS_DIR / f"{experiment_id}.json").resolve()
    if output_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing results: {output_path} (use --force)")

    load_start = time.perf_counter()
    retriever = LegalRetriever(
        qdrant_path=app_config.QDRANT_PATH,
        collection_name=app_config.COLLECTION_NAME,
        vectorizer_path=app_config.VECTORIZER_PATH,
        text_directory=app_config.TEXT_DIRECTORY,
        qdrant_url=app_config.QDRANT_URL,
        qdrant_api_key=app_config.QDRANT_API_KEY,
        embedding_model_name=app_config.EMBEDDING_MODEL_NAME,
        cross_encoder_name=config.reranker_model or app_config.CROSS_ENCODER_NAME,
        device=args.device,
        embedding_max_seq_length=config.embedding_max_seq_length,
        reranker_max_length=config.reranker_max_length,
        load_reranker=config.rerank,
    )
    startup_s = time.perf_counter() - load_start
    collection_info = collection_metadata(retriever)

    warmup_start = time.perf_counter()
    for query in queries[: args.warmup]:
        retriever.retrieve(query["query"], config)
    warmup_s = time.perf_counter() - warmup_start

    per_query: List[Dict[str, Any]] = []
    latency_rows: List[Dict[str, float]] = []
    rankings_production: Dict[str, List[str]] = {}
    rankings_top10: Dict[str, List[str]] = {}
    out_of_domain: List[Dict[str, Any]] = []

    try:
        for query in queries:
            qid = query["query_id"]
            repeats = max(1, args.repeats)
            traces = []
            for _ in range(repeats):
                traces.append(retriever.retrieve(query["query"], config))

            trace = traces[0]
            timings = {
                key: statistics.median([t["timings"][key] for t in traces])
                for key in trace["timings"]
            }
            latency_rows.append(timings)

            prod_docs = [item["file_name"] for item in trace["accepted_documents"]]
            top10_docs = [item["file_name"] for item in trace["ranked_documents"][:10]]
            rankings_production[qid] = prod_docs
            rankings_top10[qid] = top10_docs

            record = {
                "query_id": qid,
                "query": query["query"],
                "category": query.get("category"),
                "candidate_points": trace["candidate_points"],
                "candidate_documents": trace["candidate_documents"],
                "rerank_pairs": trace["rerank_pairs"],
                "production": strip_ranking(trace["accepted_documents"], config.final_top_k),
                "ranking_top10": strip_ranking(trace["ranked_documents"], 10),
                "qrels": qrels.get(qid, {}),
                "timings": {k: round(v, 6) for k, v in timings.items()},
            }
            per_query.append(record)

            if not qrels.get(qid):
                out_of_domain.append(
                    {
                        "query_id": qid,
                        "query": query["query"],
                        "production_returned": len(prod_docs),
                        "production_empty": len(prod_docs) == 0,
                        "ranking_top1": top10_docs[0] if top10_docs else None,
                        "production_file_names": prod_docs,
                    }
                )
    finally:
        retriever.close()

    production_metrics = evaluate_rankings(rankings_production, qrels)
    ranking_metrics = evaluate_rankings(rankings_top10, qrels)
    latency = latency_summary(
        latency_rows,
        ["embed_s", "sparse_s", "qdrant_s", "aggregate_s", "rerank_s", "total_s"],
    )

    # Per-category macro metrics for the top-10 ranking mode.
    category_metrics: Dict[str, Any] = {}
    for category in sorted({q.get("category", "unknown") for q in queries}):
        qids = [q["query_id"] for q in queries if q.get("category") == category]
        sub_qrels = {qid: qrels[qid] for qid in qids if qrels.get(qid)}
        if not sub_qrels:
            continue
        category_metrics[category] = evaluate_rankings(rankings_top10, sub_qrels)["metrics"]

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "sentence-transformers",
                "transformers",
                "qdrant-client",
                "scikit-learn",
                "numpy",
                "fastapi",
            )
        },
    }
    try:
        import torch

        env["torch_num_threads"] = torch.get_num_threads()
    except Exception:
        pass

    corpus_meta = {
        "pdf_count": len(list(app_config.PDF_DIRECTORY.glob("*.pdf"))),
        "text_count": len(list(app_config.TEXT_DIRECTORY.glob("*.txt"))),
        "chunk_file": str(app_config.CHUNKED_JSON),
        "chunk_file_sha256": sha256_file(app_config.CHUNKED_JSON),
        "vectorizer": str(app_config.VECTORIZER_PATH),
        "vectorizer_sha256": sha256_file(app_config.VECTORIZER_PATH),
        "collection": collection_info,
    }

    result = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "dataset": dataset_meta,
        "corpus": corpus_meta,
        "environment": env,
        "evaluation_protocol": {
            "document_level": True,
            "relevance_threshold": 1,
            "ranking_mode": "top-10 ranked documents, no score threshold",
            "production_mode": f"top-{config.final_top_k}, threshold={config.threshold}",
            "cutoffs": [1, 3, 5, 10],
            "mrr_cutoff": 10,
            "repeats_per_query": max(1, args.repeats),
            "warmup_queries": args.warmup,
        },
        "metrics_ranking_top10": ranking_metrics,
        "metrics_production": production_metrics,
        "metrics_by_category_top10": category_metrics,
        "latency_seconds": latency,
        "startup_seconds": startup_s,
        "warmup_seconds": warmup_s,
        "out_of_domain": out_of_domain,
        "per_query": per_query,
        "warnings": [
            "Relevance labels are provisional and require FYP student verification.",
            "p95 is estimated from 40 in-domain queries; it is not a production SLA.",
            "Score thresholds are sigmoid values from a general-domain reranker, not calibrated probabilities.",
            "No LLM generation was invoked; these are retrieval-only measurements.",
        ],
    }

    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    append_summary_row(result, output_path)
    print(f"Wrote {output_path}")
    print_summary(result)
    return result


def collection_metadata_from_config() -> Dict[str, Any]:
    # Metadata is recorded by the live retriever in run(); this fallback keeps
    # corpus metadata independent of an open Qdrant handle.
    meta_path = Path(app_config.QDRANT_PATH) / "meta.json"
    return {"meta_json_sha256": sha256_file(meta_path)}


def append_summary_row(result: Dict[str, Any], output_path: Path) -> None:
    ranking = result["metrics_ranking_top10"]["metrics"]
    production = result["metrics_production"]["metrics"]
    latency = result["latency_seconds"]

    def mean(block: Dict[str, Any], key: str) -> Optional[float]:
        value = (block.get(key) or {}).get("mean")
        return None if value is None else round(float(value), 6)

    row = {
        "experiment_id": result["experiment_id"],
        "timestamp_utc": result["timestamp_utc"],
        "config_name": result["config"]["name"],
        "mode": result["config"]["mode"],
        "fusion": result["config"]["fusion"],
        "prefetch_limit": result["config"]["prefetch_limit"],
        "rerank": result["config"]["rerank"],
        "rerank_max_docs": result["config"]["rerank_max_docs"],
        "max_chunks_per_doc": result["config"]["max_chunks_per_doc"],
        "include_document_level": result["config"]["include_document_level"],
        "final_top_k": result["config"]["final_top_k"],
        "threshold": result["config"]["threshold"],
        "recall@5": mean(ranking, "recall@5"),
        "recall@10": mean(ranking, "recall@10"),
        "precision@5": mean(ranking, "precision@5"),
        "mrr@10": mean(ranking, "mrr@10"),
        "map@10": mean(ranking, "map@10"),
        "ndcg@5": mean(ranking, "ndcg@5"),
        "ndcg@10": mean(ranking, "ndcg@10"),
        "production_recall@3": mean(production, "recall@3"),
        "production_ndcg@3": mean(production, "ndcg@3"),
        "median_total_s": (latency.get("total_s") or {}).get("median"),
        "p95_total_s": (latency.get("total_s") or {}).get("p95"),
        "median_rerank_s": (latency.get("rerank_s") or {}).get("median"),
        "n_queries": result["dataset"]["n_queries"],
        "result_path": str(output_path.relative_to(ROOT)),
    }
    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = COMPARISON_CSV.exists()
    with COMPARISON_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def print_summary(result: Dict[str, Any]) -> None:
    r = result["metrics_ranking_top10"]["metrics"]
    p = result["metrics_production"]["metrics"]
    lat = result["latency_seconds"]

    def fmt(block: Dict[str, Any], key: str) -> str:
        value = (block.get(key) or {}).get("mean")
        return "n/a" if value is None else f"{value:.4f}"

    print(
        "Top-10 ranking: "
        f"Recall@5={fmt(r, 'recall@5')} Recall@10={fmt(r, 'recall@10')} "
        f"MRR@10={fmt(r, 'mrr@10')} nDCG@10={fmt(r, 'ndcg@10')}"
    )
    print(
        "Production: "
        f"Recall@3={fmt(p, 'recall@3')} nDCG@3={fmt(p, 'ndcg@3')} "
        f"P@3={fmt(p, 'precision@3')}"
    )
    total = lat.get("total_s") or {}
    if total:
        print(
            f"Latency: median={total['median']:.3f}s p95={total['p95']:.3f}s "
            f"(n={lat['n_queries']})"
        )
    print(
        f"Out-of-domain: {sum(1 for q in result['out_of_domain'] if not q['production_empty'])}"
        f"/{len(result['out_of_domain'])} returned a non-empty production result"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to retrieval config JSON")
    parser.add_argument("--queries", default=str(QUERIES_PATH))
    parser.add_argument("--qrels", default=str(QRELS_PATH))
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--name", default=None, help="Override config name")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
