"""실험 실행 — config를 받아 평가셋 전체를 돌리고 결과표를 results/에 누적한다.

같은 qa_set.jsonl으로 한 변수씩 바꿔(config 오버라이드) 비교하는 것이 목적이다.

사용법:
    # baseline(default.yaml)으로 검색 채점
    python scripts/run_experiment.py --label baseline

    # 실험1: 검색 방식 비교 (config 오버라이드)
    python scripts/run_experiment.py --config config/experiments/exp1_retrieval.yaml --label exp1-sparse

    # 답변 품질까지 채점(로컬 LLM 구동 필요)
    python scripts/run_experiment.py --label baseline --judge

결과:
    results/<label>_per_question.csv   질문별 상세
    results/summary.md                 실험별 요약표(append 누적)
    results/summary.csv                실험별 요약표(append 누적)
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import evaluation, generation, retrieval
from src.config import load_config, set_seed

K_VALUES = (1, 3, 5)


def _judge_answers(config, eval_set, per_question):
    """질문별로 답변을 생성하고 로컬 LLM 심판으로 채점한다(가용 시)."""
    if not generation.is_backend_available(config):
        print("[안내] 로컬 LLM 미구동 — 답변 품질 채점을 건너뜁니다.")
        return None

    by_id = {q["id"]: q for q in eval_set}
    scores: list[float] = []
    for row in per_question:
        item = by_id[row["id"]]
        results = retrieval.retrieve(config, item["question"])
        context = [r["text"] for r in results]
        prompt = generation.build_answer_prompt(item["question"], context)
        prediction = generation.generate(config, prompt)
        judged = evaluation.judge_answer(config, item["question"], item["answer"], prediction)
        row["correctness"] = judged["correctness"]
        if judged["correctness"] is not None:
            scores.append(judged["correctness"])
    return sum(scores) / len(scores) if scores else None


def _write_per_question(results_dir: Path, label: str, per_question: list[dict]) -> Path:
    path = results_dir / f"{label}_per_question.csv"
    fields = ["id", "type", "question", "gold", "retrieved", "rr"]
    fields += [f"hit@{k}" for k in K_VALUES]
    if per_question and "correctness" in per_question[0]:
        fields.append("correctness")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in per_question:
            out = dict(row)
            out["gold"] = "|".join(row["gold"])
            out["retrieved"] = "|".join(row["retrieved"])
            writer.writerow(out)
    return path


def _append_summary(results_dir: Path, label: str, config, summary: dict, answer_score) -> None:
    """실험별 요약을 summary.md / summary.csv에 누적(append)한다."""
    ov = summary["overall"]
    r = config["retrieval"]
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "label": label,
        "n": int(ov["n"]),
        "mode": r.get("mode", ""),
        "reranker": r["reranker"]["enabled"],
        "embedding": config["embedding"]["model"].split("/")[-1],
        "mrr": round(ov["mrr"], 4),
        "hit@1": round(ov["hit@1"], 4),
        "hit@3": round(ov["hit@3"], 4),
        "hit@5": round(ov["hit@5"], 4),
        "answer": "" if answer_score is None else round(answer_score, 4),
    }

    csv_path = results_dir / "summary.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    md_path = results_dir / "summary.md"
    if not md_path.exists():
        header = "| " + " | ".join(row.keys()) + " |\n"
        sep = "| " + " | ".join(["---"] * len(row)) + " |\n"
        md_path.write_text("# 실험 결과 요약\n\n" + header + sep, encoding="utf-8")
    line = "| " + " | ".join(str(v) for v in row.values()) + " |\n"
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="평가셋 전체 실행 + 결과표 누적")
    parser.add_argument("--config", default=None, help="실험 오버라이드 YAML 경로")
    parser.add_argument("--label", required=True, help="결과표에 기록할 실험 이름")
    parser.add_argument("--judge", action="store_true", help="답변 품질까지 채점(로컬 LLM 필요)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    eval_set = evaluation.load_eval_set(config["paths"]["eval_path"])
    print(f"[평가셋] {len(eval_set)}문항 로드 — 검색 채점 시작 (label={args.label})")

    summary = evaluation.evaluate_retrieval(config, eval_set, K_VALUES)

    answer_score = None
    if args.judge:
        answer_score = _judge_answers(config, eval_set, summary["per_question"])

    # 결과 출력
    ov = summary["overall"]
    print(f"\n[검색 채점 — 전체 {int(ov['n'])}문항]")
    print(f"  MRR    = {ov['mrr']:.4f}")
    for k in K_VALUES:
        print(f"  Hit@{k}  = {ov[f'hit@{k}']:.4f}")
    print("\n[질문 유형별]")
    for t, agg in summary["by_type"].items():
        print(f"  {t:<8} (n={int(agg['n'])}): MRR={agg['mrr']:.4f}  "
              f"Hit@1={agg['hit@1']:.4f}  Hit@3={agg['hit@3']:.4f}  Hit@5={agg['hit@5']:.4f}")
    if answer_score is not None:
        print(f"\n[답변 품질] 평균 정확성(로컬 LLM 심판) = {answer_score:.4f}")

    # 결과 파일 기록
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    pq_path = _write_per_question(results_dir, args.label, summary["per_question"])
    _append_summary(results_dir, args.label, config, summary, answer_score)
    print(f"\n[저장] 질문별 상세: {pq_path}")
    print(f"[저장] 요약 누적: {results_dir/'summary.md'}, {results_dir/'summary.csv'}")


if __name__ == "__main__":
    main()
