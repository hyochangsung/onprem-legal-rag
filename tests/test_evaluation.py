"""평가 단위 테스트 — Hit Rate@k / MRR 순수 계산 + 점수 파싱(모델 불필요)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import _aggregate, _parse_score, hit_at_k, reciprocal_rank


def test_hit_at_k_basic():
    ranked = ["jo-3", "jo-1", "jo-7"]
    gold = {"jo-1"}
    assert hit_at_k(ranked, gold, 1) == 0.0   # 1위는 jo-3 → 미적중
    assert hit_at_k(ranked, gold, 2) == 1.0   # 2위 안에 jo-1 → 적중
    assert hit_at_k(ranked, gold, 5) == 1.0


def test_hit_at_k_multi_gold():
    ranked = ["jo-9", "jo-13", "jo-2"]
    gold = {"jo-7", "jo-13"}  # 둘 중 하나만 들어와도 적중
    assert hit_at_k(ranked, gold, 1) == 0.0
    assert hit_at_k(ranked, gold, 2) == 1.0


def test_reciprocal_rank():
    gold = {"jo-1"}
    assert reciprocal_rank(["jo-1", "jo-2"], gold) == 1.0          # 1위
    assert reciprocal_rank(["jo-2", "jo-1"], gold) == 0.5          # 2위
    assert reciprocal_rank(["jo-2", "jo-3", "jo-1"], gold) == 1 / 3
    assert reciprocal_rank(["jo-2", "jo-3"], gold) == 0.0          # 정답 없음


def test_aggregate_means():
    rows = [
        {"rr": 1.0, "hit@1": 1.0, "hit@3": 1.0},
        {"rr": 0.0, "hit@1": 0.0, "hit@3": 0.0},
    ]
    agg = _aggregate(rows, (1, 3))
    assert agg["n"] == 2.0
    assert agg["mrr"] == 0.5
    assert agg["hit@1"] == 0.5
    assert agg["hit@3"] == 0.5


def test_parse_score_normalization():
    assert _parse_score("0.8") == 0.8
    assert _parse_score("점수: 1.0") == 1.0
    assert _parse_score("8") == 0.8        # 0~10 스케일 보정
    assert _parse_score("85점") == 0.85    # 0~100 스케일 보정
    assert _parse_score("설명만 있고 숫자 없음") is None
    assert _parse_score("2.0") == 0.2      # 1<값<=10 → 0~10 스케일 보정
    assert _parse_score("150점") == 1.0    # 0~100 스케일 보정 후 상한 클램프


if __name__ == "__main__":
    test_hit_at_k_basic()
    test_hit_at_k_multi_gold()
    test_reciprocal_rank()
    test_aggregate_means()
    test_parse_score_normalization()
    print("모든 테스트 통과")
