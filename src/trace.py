"""파이프라인 추적 — 검색 과정의 모든 중간 단계를 구조화해 반환한다.

목적: 챗봇 데모 화면에서 "질문 → 임베딩 → 벡터DB 매칭 → 프롬프트 → LLM 답변"의
전 과정을 한 페이지에 투명하게 보여주기 위해, 기존 retrieval.retrieve()가 내부적으로만
쓰던 중간 결과(질문 임베딩 벡터, dense/sparse 후보와 점수, RRF 결합, 리랭킹, 최종 조)를
밖으로 드러낸다.

주의: 검색 로직 자체는 retrieval 모듈의 내부 함수를 재사용한다. 여기서는 점수·순위 등
"보여줄 정보"를 추가로 수집할 뿐, 검색 방식을 새로 정의하지 않는다(설계 일관성 유지).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src import indexing, retrieval


def _snippet(text: str, n: int = 140) -> str:
    """긴 조문 텍스트를 카드에 보여줄 길이로 줄인다."""
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + " …"


def _build_text_lookup(config: dict[str, Any]) -> dict[str, str]:
    """검색 단위 id → 텍스트 매핑(BM25 영속화 데이터 재사용)."""
    bm = retrieval._load_bm25(config)
    return {uid: doc for uid, doc in zip(bm["ids"], bm["docs"])}


def _dense_hits(config, qemb, k, text_lookup):
    """벡터 검색 후보를 점수와 함께 수집한다(점수를 버리지 않음).

    질문 임베딩(qemb)은 2단계에서 1회만 계산해 넘겨받는다(중복 인코딩 방지).
    """
    client = retrieval._get_qdrant_client(config)
    points = client.query_points(
        collection_name=config["indexing"]["collection_name"],
        query=qemb,
        limit=k,
        with_payload=True,
    ).points
    hits = []
    for rank, h in enumerate(points):
        meta = h.payload or {}
        doc_id = meta.get("doc_id", str(h.id))
        hits.append(
            {
                "rank": rank + 1,
                "doc_id": doc_id,
                "parent_id": meta.get("parent_id", doc_id),
                "jo": meta.get("jo", ""),
                "jo_title": meta.get("jo_title", ""),
                "score": round(float(h.score), 4),
                "snippet": _snippet(text_lookup.get(doc_id, "")),
            }
        )
    return hits


def _sparse_hits(config, query, k, text_lookup):
    """BM25 검색 후보를 점수·질문 토큰과 함께 수집한다."""
    bm = retrieval._load_bm25(config)
    tokens = indexing.tokenize_ko(query)
    scores = bm["bm25"].get_scores(tokens)
    order = np.argsort(scores)[::-1][:k]
    hits = []
    for rank, i in enumerate(order):
        meta = bm["metadatas"][i]
        doc_id = bm["ids"][i]
        hits.append(
            {
                "rank": rank + 1,
                "doc_id": doc_id,
                "parent_id": meta.get("parent_id", doc_id),
                "jo": meta.get("jo", ""),
                "jo_title": meta.get("jo_title", ""),
                "score": round(float(scores[i]), 4),
                "snippet": _snippet(text_lookup.get(doc_id, "")),
            }
        )
    return tokens, hits


def trace(config: dict[str, Any], query: str) -> dict[str, Any]:
    """질문 하나에 대한 전체 검색 파이프라인 추적 결과를 반환한다.

    반환 구조(프런트엔드가 그대로 단계별 카드로 렌더):
      {
        question, embedding{...}, dense{...}, sparse{...},
        rrf{...}, reranker{...}, final[...], prompt
      }
    """
    r = config["retrieval"]
    mode = r.get("mode", "hybrid")
    cand_k = r["candidate_k"]
    top_k = r["top_k"]

    # 단계별 소요 시간(초)을 누적. 화면에서 각 단계 카드에 표시한다.
    timings: dict[str, float] = {}
    t_total = time.perf_counter()

    _t = time.perf_counter()
    text_lookup = _build_text_lookup(config)
    resolve = retrieval._build_parent_resolver(config)
    timings["load"] = time.perf_counter() - _t  # 색인(BM25/parent) 로드

    out: dict[str, Any] = {"question": query, "mode": mode,
                           "top_k": top_k, "candidate_k": cand_k}

    # ── 1) 질문 임베딩 ──────────────────────────────────────────────
    _t = time.perf_counter()
    qemb = indexing.encode_queries(config, [query])[0]
    timings["embedding"] = time.perf_counter() - _t
    vec = np.asarray(qemb, dtype=float)
    out["embedding"] = {
        "model": config["embedding"]["model"],
        "query_prefix": indexing._QUERY_PREFIX,
        "prefixed_text": indexing._QUERY_PREFIX + query,
        "dim": len(qemb),
        "normalized": bool(config["embedding"]["normalize"]),
        "norm": round(float(np.linalg.norm(vec)), 4),
        "preview": [round(float(x), 4) for x in vec[:48]],  # 앞 48차원 미리보기
    }

    # ── 2) dense / sparse 후보 ─────────────────────────────────────
    dense_hits: list[dict] = []
    sparse_hits: list[dict] = []
    dense_pids: list[str] = []
    sparse_pids: list[str] = []

    if mode in ("dense", "hybrid"):
        _t = time.perf_counter()
        dense_hits = _dense_hits(config, qemb, cand_k, text_lookup)
        timings["dense_search"] = time.perf_counter() - _t  # 임베딩 제외, 벡터 탐색만
        dense_pids = retrieval._to_parent_ranking(
            [(h["doc_id"], {"parent_id": h["parent_id"]}) for h in dense_hits]
        )
    if mode in ("sparse", "hybrid"):
        _t = time.perf_counter()
        tokens, sparse_hits = _sparse_hits(config, query, cand_k, text_lookup)
        timings["sparse_search"] = time.perf_counter() - _t  # 토큰화 + BM25 점수
        out["sparse_query_tokens"] = tokens
        sparse_pids = retrieval._to_parent_ranking(
            [(h["doc_id"], {"parent_id": h["parent_id"]}) for h in sparse_hits]
        )

    out["dense"] = {"hits": dense_hits[:10], "n": len(dense_hits)}
    out["sparse"] = {
        "hits": sparse_hits[:10],
        "n": len(sparse_hits),
        "tokens": out.get("sparse_query_tokens", []),
    }

    # ── 3) RRF 결합 ────────────────────────────────────────────────
    def _jo_label(pid: str) -> tuple[str, str]:
        item = resolve(pid)
        if not item:
            return pid, ""
        m = item["metadata"]
        return m.get("jo", pid), m.get("jo_title", "")

    if mode == "hybrid":
        _t = time.perf_counter()
        rankings = {"dense": dense_pids, "sparse": sparse_pids}
        combined = retrieval.rrf_combine(rankings, r["weights"], r["rrf_k"])
        timings["rrf"] = time.perf_counter() - _t
        ranked_pids = [pid for pid, _ in combined]
        scores = dict(combined)
        dense_rank = {pid: i + 1 for i, pid in enumerate(dense_pids)}
        sparse_rank = {pid: i + 1 for i, pid in enumerate(sparse_pids)}
        rrf_rows = []
        for pid, sc in combined[:10]:
            jo, jo_title = _jo_label(pid)
            rrf_rows.append(
                {
                    "parent_id": pid,
                    "jo": jo,
                    "jo_title": jo_title,
                    "dense_rank": dense_rank.get(pid),
                    "sparse_rank": sparse_rank.get(pid),
                    "score": round(float(sc), 5),
                }
            )
        out["rrf"] = {
            "applied": True,
            "weights": r["weights"],
            "rrf_k": r["rrf_k"],
            "formula": "score(조) = Σ_method  w · 1 / (rrf_k + rank)",
            "rows": rrf_rows,
        }
    else:
        single = "dense" if mode == "dense" else "sparse"
        ranked_pids = dense_pids if mode == "dense" else sparse_pids
        scores = {pid: 1.0 / (i + 1) for i, pid in enumerate(ranked_pids)}
        out["rrf"] = {"applied": False, "reason": f"{single} 단독 모드 — 결합 단계 없음"}

    # ── 4) 리랭커(옵션) ────────────────────────────────────────────
    rr = r["reranker"]
    order_before = list(ranked_pids)
    reranker_applied = False
    if rr.get("enabled") and rr.get("model"):
        def parent_text(pid: str) -> str:
            item = resolve(pid)
            return item["text"] if item else ""

        _t = time.perf_counter()
        ranked_pids = retrieval._rerank(config, query, ranked_pids, parent_text)
        timings["rerank"] = time.perf_counter() - _t
        reranker_applied = True

    def _label_list(pids):
        rows = []
        for i, pid in enumerate(pids[:top_k]):
            jo, jo_title = _jo_label(pid)
            rows.append({"rank": i + 1, "parent_id": pid, "jo": jo, "jo_title": jo_title})
        return rows

    out["reranker"] = {
        "enabled": bool(rr.get("enabled")),
        "model": rr.get("model", ""),
        "applied": reranker_applied,
        "note": ("" if reranker_applied else
                 ("모델 미지정 — 재정렬 건너뜀(원순서 유지)"
                  if rr.get("enabled") else "비활성화")),
        "order_before": _label_list(order_before),
        "order_after": _label_list(ranked_pids),
    }

    # ── 5) 최종 조(parent) 조립 ────────────────────────────────────
    final = []
    for pid in ranked_pids[:top_k]:
        item = resolve(pid)
        if item is None:
            continue
        m = item["metadata"]
        final.append(
            {
                "id": pid,
                "jo": m.get("jo", pid),
                "jo_title": m.get("jo_title", ""),
                "path": m.get("path", ""),
                "score": round(float(scores.get(pid, 0.0)), 5),
                "text": item["text"],
            }
        )
    out["final"] = final

    # ── 6) LLM 프롬프트 구성 ───────────────────────────────────────
    from src import generation

    _t = time.perf_counter()
    out["prompt"] = generation.build_answer_prompt(query, [f["text"] for f in final])
    timings["prompt"] = time.perf_counter() - _t

    # 검색(2~4단계) 합계 = 임베딩 + dense + sparse + rrf + rerank (로드 제외)
    timings["retrieval_total"] = sum(
        timings.get(k, 0.0)
        for k in ("embedding", "dense_search", "sparse_search", "rrf", "rerank")
    )
    timings["total"] = time.perf_counter() - t_total
    # 초 단위, 소수 4자리로 반올림해 전달
    out["timings"] = {k: round(v, 4) for k, v in timings.items()}
    return out
