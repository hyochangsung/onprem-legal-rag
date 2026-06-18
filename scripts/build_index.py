"""인덱스 빌드 엔트리포인트 — 청킹 + 인덱싱을 1회 실행한다.

사용법:
    python scripts/build_index.py                                   # default 설정
    python scripts/build_index.py --config config/experiments/exp4_embedding.yaml

청킹 방식·임베딩 모델 등 인덱스에 영향을 주는 변수를 바꾼 실험에서는
이 스크립트로 인덱스를 다시 빌드해야 한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import chunking, indexing
from src.config import load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="청킹 + 인덱싱 빌드")
    parser.add_argument(
        "--config",
        default=None,
        help="실험 오버라이드 YAML 경로 (없으면 default.yaml만 사용)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    print("[1/2] 청킹...")
    chunks = chunking.run(config)
    n_parent = sum(1 for c in chunks if c["type"] == "parent")
    n_child = sum(1 for c in chunks if c["type"] == "child")
    n_article = sum(1 for c in chunks if c["type"] == "article")
    print(f"  청크 {len(chunks)}개 (parent={n_parent}, child={n_child}, article={n_article})")

    print("[2/2] 인덱싱...")
    stats = indexing.run(config)
    print(f"  검색 단위={stats['search_units']}, parent lookup={stats['parents']}")
    print("완료. 벡터 색인:", config["paths"]["vectorstore_dir"],
          "| BM25:", config["paths"]["bm25_path"])


if __name__ == "__main__":
    main()
