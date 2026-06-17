"""평가 — 검색 채점(Hit Rate@k, MRR) + (옵션) 답변 품질 채점.

설계 원칙(CLAUDE.md):
  - 모든 실험은 동일 질문셋(data/eval/qa_set.jsonl)으로 비교한다.
  - 검색 채점은 정답 조(jo) id 기준. retrieve()가 조(parent) id를 반환하므로,
    질문별 정답 조 id 집합과 검색 결과 순위를 비교해 Hit Rate@k / MRR을 계산한다.
  - 질문 유형별(일상어 / 조항용어)로도 분리 집계한다.
  - 답변 품질: 온프레미스 제약상 외부 API 금지. 로컬 LLM(Ollama)을 심판으로 쓰는
    경량 채점을 제공하며, 백엔드가 없으면 건너뛴다. (RAGAS 정식 연동은 로컬 LLM·임베딩
    배선이 추가로 필요하므로 후속 작업으로 둔다.)

검색 채점 함수(hit_at_k/reciprocal_rank)는 모델 없이 순수 계산이라 단위 테스트 가능.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 평가셋 로드
# ---------------------------------------------------------------------------
def load_eval_set(path: str | Path) -> list[dict[str, Any]]:
    """질문-정답 평가셋(jsonl)을 로드한다.

    각 항목: {id, type, question, gold_jo(list[str]), answer(str)}
    """
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# 검색 채점 지표 (순수 계산)
# ---------------------------------------------------------------------------
def hit_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    """상위 k개 안에 정답 조가 하나라도 있으면 1.0, 없으면 0.0."""
    return 1.0 if any(pid in gold for pid in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: list[str], gold: set[str]) -> float:
    """첫 정답 조의 역순위(1/rank). 정답이 전혀 없으면 0.0."""
    for rank, pid in enumerate(ranked_ids, start=1):
        if pid in gold:
            return 1.0 / rank
    return 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# 검색 채점 실행
# ---------------------------------------------------------------------------
def evaluate_retrieval(
    config: dict[str, Any],
    eval_set: list[dict[str, Any]],
    k_values: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    """평가셋 전체에 대해 검색을 돌리고 Hit Rate@k / MRR을 집계한다.

    Hit Rate@k 계산을 위해 최소 max(k) 개의 순위가 필요하므로, 이 실행에 한해
    top_k를 max(k_values)로 올린 config 사본을 사용한다(순위 자체는 변하지 않음).

    Returns:
        {
          "overall": {"n":..., "mrr":..., "hit@1":..., ...},
          "by_type": {"일상어": {...}, "조항용어": {...}},
          "per_question": [{"id","type","gold","retrieved","rr","hit@k"...}, ...],
        }
    """
    from copy import deepcopy

    from src import retrieval

    max_k = max(k_values)
    run_config = deepcopy(config)
    run_config["retrieval"]["top_k"] = max(run_config["retrieval"].get("top_k", max_k), max_k)

    per_question: list[dict[str, Any]] = []
    for item in eval_set:
        gold = set(item["gold_jo"])
        results = retrieval.retrieve(run_config, item["question"])
        ranked_ids = [r["id"] for r in results]

        row: dict[str, Any] = {
            "id": item["id"],
            "type": item.get("type", ""),
            "question": item["question"],
            "gold": sorted(gold),
            "retrieved": ranked_ids[:max_k],
            "rr": reciprocal_rank(ranked_ids, gold),
        }
        for k in k_values:
            row[f"hit@{k}"] = hit_at_k(ranked_ids, gold, k)
        per_question.append(row)

    overall = _aggregate(per_question, k_values)
    by_type: dict[str, Any] = {}
    types = sorted({r["type"] for r in per_question if r["type"]})
    for t in types:
        rows = [r for r in per_question if r["type"] == t]
        by_type[t] = _aggregate(rows, k_values)

    return {"overall": overall, "by_type": by_type, "per_question": per_question}


def _aggregate(rows: list[dict[str, Any]], k_values: tuple[int, ...]) -> dict[str, float]:
    """질문별 행 리스트를 평균 지표로 집계한다."""
    agg: dict[str, float] = {"n": float(len(rows)), "mrr": _mean([r["rr"] for r in rows])}
    for k in k_values:
        agg[f"hit@{k}"] = _mean([r[f"hit@{k}"] for r in rows])
    return agg


# ---------------------------------------------------------------------------
# 답변 품질 채점 (옵션, 로컬 LLM 심판)
# ---------------------------------------------------------------------------
def _parse_score(text: str) -> float | None:
    """LLM 응답에서 0~1 점수를 추출한다. 실패 시 None."""
    import re

    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        score = float(m.group(1))
    except ValueError:
        return None
    # 0~1 범위로 보정(모델이 0~100 또는 0~10으로 답하는 경우)
    if score > 1.0:
        score = score / 100.0 if score > 10.0 else score / 10.0
    return max(0.0, min(1.0, score))


def judge_answer(
    config: dict[str, Any], question: str, gold_answer: str, prediction: str
) -> dict[str, float | None]:
    """로컬 LLM을 심판으로 답변의 정확성·근거성을 0~1로 채점한다.

    온프레미스 제약상 외부 API를 쓰지 않고 로컬 LLM(Ollama)만 사용한다.
    백엔드가 없으면 {"correctness": None, ...}를 반환해 채점을 건너뛴다.
    """
    from src import generation

    if not generation.is_backend_available(config):
        return {"correctness": None}

    prompt = (
        "당신은 사내 규정 QA 시스템의 답변을 채점하는 평가자입니다. "
        "아래 [질문]에 대한 [기준 답변]과 [시스템 답변]을 비교하여, 시스템 답변이 "
        "기준 답변의 핵심 사실과 얼마나 일치하는지 0.0부터 1.0 사이 숫자 하나로만 답하세요. "
        "설명 없이 숫자만 출력하세요.\n\n"
        f"[질문]\n{question}\n\n"
        f"[기준 답변]\n{gold_answer}\n\n"
        f"[시스템 답변]\n{prediction}\n\n[점수]\n"
    )
    raw = generation.generate(config, prompt)
    return {"correctness": _parse_score(raw)}
