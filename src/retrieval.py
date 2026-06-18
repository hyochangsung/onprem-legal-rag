"""검색 — Hybrid(벡터 + BM25) + RRF 결합 + (옵션) 메타필터 + (옵션) 리랭커.

설계 원칙(CLAUDE.md):
  - 항(child) 단위로 검색하되, 최종적으로 조(parent) 전체를 반환해 LLM에 전달.
  - 벡터/BM25 후보를 RRF로 결합(점수가 아닌 순위로 통합 → 스케일 차이 무관).
  - RRF sparse/dense 가중치, top_k, 리랭커 on/off는 모두 config(실험1·3).

처리 순서:
  질문 → 벡터 검색 + BM25 검색 → (조 단위 순위로 환원) → RRF 결합
       → (옵션) cross-encoder 리랭크 → 조(parent) 전체로 확장 → 상위 top_k 반환
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src import indexing

# 리랭커 모델 미지정 경고를 1회만 출력하기 위한 플래그
_warned_reranker = [False]


# ---------------------------------------------------------------------------
# 색인 로드
# ---------------------------------------------------------------------------
def _load_bm25(config: dict[str, Any]) -> dict[str, Any]:
    with open(config["paths"]["bm25_path"], "rb") as f:
        return pickle.load(f)


def _load_parents(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = Path(config["paths"]["vectorstore_dir"]) / "parents.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# Qdrant 로컬 모드는 디스크 락을 잡으므로, 클라이언트를 프로세스당 1개만 열어 재사용한다.
_qdrant_client_cache: dict[str, Any] = {}


def _get_qdrant_client(config: dict[str, Any]):
    from qdrant_client import QdrantClient

    persist_dir = config["paths"]["vectorstore_dir"]
    if persist_dir not in _qdrant_client_cache:
        _qdrant_client_cache[persist_dir] = QdrantClient(path=persist_dir)
    return _qdrant_client_cache[persist_dir]


def _build_qdrant_filter(where: dict | None):
    if not where:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    return Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in where.items()])


# ---------------------------------------------------------------------------
# 개별 검색
# ---------------------------------------------------------------------------
def _vector_search(
    config: dict[str, Any], query: str, k: int, where: dict | None
) -> list[tuple[str, dict]]:
    """벡터 검색 결과를 (search_unit_id, metadata) 순위 리스트로 반환."""
    client = _get_qdrant_client(config)
    qemb = indexing.encode_queries(config, [query])[0]
    hits = client.query_points(
        collection_name=config["indexing"]["collection_name"],
        query=qemb,
        limit=k,
        query_filter=_build_qdrant_filter(where),
        with_payload=True,
    ).points
    return [(str(h.id), h.payload) for h in hits]


def _bm25_search(
    config: dict[str, Any], query: str, k: int, where: dict | None
) -> list[tuple[str, dict]]:
    """BM25 검색 결과를 (search_unit_id, metadata) 순위 리스트로 반환."""
    bm = _load_bm25(config)
    tokens = indexing.tokenize_ko(query)
    scores = bm["bm25"].get_scores(tokens)
    order = np.argsort(scores)[::-1]
    hits: list[tuple[str, dict]] = []
    for i in order:
        meta = bm["metadatas"][i]
        if where and not _meta_match(meta, where):
            continue
        hits.append((bm["ids"][i], meta))
        if len(hits) >= k:
            break
    return hits


def _meta_match(meta: dict, where: dict) -> bool:
    """단순 동등 비교 메타데이터 필터(BM25 후처리용)."""
    return all(meta.get(key) == val for key, val in where.items())


# ---------------------------------------------------------------------------
# 조(parent) 단위 순위 환원 + RRF
# ---------------------------------------------------------------------------
def _to_parent_ranking(hits: list[tuple[str, dict]]) -> list[str]:
    """검색 히트(항)를 조(parent_id) 순위로 환원(첫 등장=최고 순위)."""
    seen: set[str] = set()
    ranking: list[str] = []
    for _uid, meta in hits:
        pid = meta["parent_id"]
        if pid not in seen:
            seen.add(pid)
            ranking.append(pid)
    return ranking


def rrf_combine(
    rankings: dict[str, list[str]], weights: dict[str, float], rrf_k: int
) -> list[tuple[str, float]]:
    """RRF로 여러 순위 리스트를 결합한다.

    각 방식에서 parent_id의 순위(rank, 0-based)에 대해 weight * 1/(rrf_k + rank + 1)를
    더해 최종 점수를 매긴다.
    """
    scores: dict[str, float] = defaultdict(float)
    for method, ranking in rankings.items():
        w = weights.get(method, 1.0)
        for rank, pid in enumerate(ranking):
            scores[pid] += w * (1.0 / (rrf_k + rank + 1))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# 리랭커
# ---------------------------------------------------------------------------
_ce_cache: dict[str, Any] = {}


def _rerank(
    config: dict[str, Any], query: str, parent_ids: list[str], parent_text
) -> list[str]:
    """cross-encoder로 조 후보를 재정렬한다. 모델 미지정 시 원순서 유지."""
    rr = config["retrieval"]["reranker"]
    model_name = rr.get("model", "")
    if not model_name:
        if not _warned_reranker[0]:
            print("[경고] 리랭커 모델 미지정 — 재정렬을 건너뜁니다. "
                  "config retrieval.reranker.model 을 지정하세요.")
            _warned_reranker[0] = True
        return parent_ids

    from sentence_transformers import CrossEncoder

    if model_name not in _ce_cache:
        _ce_cache[model_name] = CrossEncoder(model_name)
    ce = _ce_cache[model_name]

    pairs = [(query, parent_text(pid)) for pid in parent_ids]
    scores = ce.predict(pairs)
    order = np.argsort(scores)[::-1]
    return [parent_ids[i] for i in order]


# ---------------------------------------------------------------------------
# 조(parent) 전체로 확장
# ---------------------------------------------------------------------------
def _build_parent_resolver(config: dict[str, Any]):
    """parent_id → (조 전체 텍스트, 메타데이터) 해소 함수.

    parent_child 모드: parents.json 사용.
    single 모드: parents.json이 비어 있으므로 검색 단위(article) 자체가 곧 조.
    """
    parents = _load_parents(config)
    bm = _load_bm25(config)
    unit_lookup = {
        uid: {"text": doc, "metadata": meta}
        for uid, doc, meta in zip(bm["ids"], bm["docs"], bm["metadatas"])
    }

    def resolve(pid: str) -> dict[str, Any] | None:
        if pid in parents:
            return parents[pid]
        if pid in unit_lookup:
            return unit_lookup[pid]
        return None

    return resolve


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def retrieve(
    config: dict[str, Any], query: str, where: dict | None = None
) -> list[dict[str, Any]]:
    """질문에 대해 최종 조(parent) 리스트를 상위 top_k개 반환한다.

    Args:
        where: 메타데이터 필터(예: {"jang": "제2장 ..."}). config retrieval.metadata_filter가
               True이고 where가 주어졌을 때만 적용.

    Returns:
        [{"id": 조id, "jo": "제N조", "text": 조전체, "metadata": {...}, "score": float}, ...]
    """
    r = config["retrieval"]
    mode = r.get("mode", "hybrid")
    cand_k = r["candidate_k"]
    use_filter = r.get("metadata_filter", False)
    eff_where = where if (use_filter and where) else None

    rankings: dict[str, list[str]] = {}
    if mode in ("dense", "hybrid"):
        rankings["dense"] = _to_parent_ranking(
            _vector_search(config, query, cand_k, eff_where)
        )
    if mode in ("sparse", "hybrid"):
        rankings["sparse"] = _to_parent_ranking(
            _bm25_search(config, query, cand_k, eff_where)
        )

    # 결합
    if mode == "hybrid":
        combined = rrf_combine(rankings, r["weights"], r["rrf_k"])
        ranked_pids = [pid for pid, _ in combined]
        scores = dict(combined)
    else:
        single = "dense" if mode == "dense" else "sparse"
        ranked_pids = rankings[single]
        scores = {pid: 1.0 / (i + 1) for i, pid in enumerate(ranked_pids)}

    # 조 전체로 확장 (리랭커가 조 텍스트를 사용하므로 먼저 resolver 준비)
    resolve = _build_parent_resolver(config)

    def parent_text(pid: str) -> str:
        item = resolve(pid)
        return item["text"] if item else ""

    # 리랭크(옵션) — 후보를 재정렬
    if r["reranker"]["enabled"]:
        ranked_pids = _rerank(config, query, ranked_pids, parent_text)

    # 상위 top_k 조립
    results: list[dict[str, Any]] = []
    for pid in ranked_pids[: r["top_k"]]:
        item = resolve(pid)
        if item is None:
            continue
        results.append(
            {
                "id": pid,
                "jo": item["metadata"].get("jo", pid),
                "text": item["text"],
                "metadata": item["metadata"],
                "score": float(scores.get(pid, 0.0)),
            }
        )
    return results
