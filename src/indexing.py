"""인덱싱 — 벡터 색인(Qdrant, 로컬 파일 임베디드 모드) + BM25 통계.

설계 원칙(CLAUDE.md):
  - "Hybrid"는 검색 시점의 결합 방식. 인덱싱은 그 Hybrid가 쓸 색인 2개를 각각 준비한다.
      1) 벡터 색인 : 검색 단위(항/조)를 임베딩해 Qdrant에 저장 (서버 없이 path=로 디스크에 영속화)
      2) BM25 통계 : 검색 단위를 한국어 토큰화해 코퍼스 통계를 1회 계산 후 영속화
  - parent(조 전체)는 검색 대상이 아니라, 검색된 child를 조 전체로 확장하기 위한 lookup으로 저장.

질문과 문서는 반드시 동일한 임베딩 모델로 변환해야 하므로, 임베딩 함수는
retrieval 단계에서도 재사용할 수 있도록 여기서 제공한다(encode_documents/encode_queries).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 임베딩 (문서/질문 공용)
# ---------------------------------------------------------------------------
# arctic-embed 계열은 질문(query)에만 프리픽스를 권장한다. 문서는 그대로 인코딩.
_QUERY_PREFIX = "query: "

_embedder_cache: dict[str, Any] = {}


def get_embedder(config: dict[str, Any]):
    """SentenceTransformer 임베더를 로드(캐시)한다."""
    from sentence_transformers import SentenceTransformer

    model_name = config["embedding"]["model"]
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = SentenceTransformer(model_name, trust_remote_code=True)
    return _embedder_cache[model_name]


def encode_documents(config: dict[str, Any], texts: list[str]) -> list[list[float]]:
    model = get_embedder(config)
    emb = model.encode(
        texts,
        batch_size=config["embedding"]["batch_size"],
        normalize_embeddings=config["embedding"]["normalize"],
        show_progress_bar=False,
    )
    return emb.tolist()


def encode_queries(config: dict[str, Any], queries: list[str]) -> list[list[float]]:
    model = get_embedder(config)
    prefixed = [_QUERY_PREFIX + q for q in queries]
    emb = model.encode(
        prefixed,
        batch_size=config["embedding"]["batch_size"],
        normalize_embeddings=config["embedding"]["normalize"],
        show_progress_bar=False,
    )
    return emb.tolist()


# ---------------------------------------------------------------------------
# 한국어 토큰화 (BM25용)
# ---------------------------------------------------------------------------
_kiwi_cache: list[Any] = []


def tokenize_ko(text: str) -> list[str]:
    """kiwipiepy 형태소 분석으로 토큰(형태소 표면형) 리스트를 반환한다."""
    if not _kiwi_cache:
        from kiwipiepy import Kiwi

        _kiwi_cache.append(Kiwi())
    kiwi = _kiwi_cache[0]
    return [tok.form for tok in kiwi.tokenize(text)]


# ---------------------------------------------------------------------------
# 청크 로드 / 분리
# ---------------------------------------------------------------------------
def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def split_search_and_parents(
    chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """검색 단위와 parent lookup을 분리한다.

    - parent_child 모드: child가 검색 단위, parent는 lookup.
    - single 모드: article이 검색 단위, parent lookup은 비어 있음(자기 자신이 곧 조).
    """
    search_units = [c for c in chunks if c["type"] in ("child", "article")]
    parents = {
        c["id"]: {"text": c["text"], "metadata": c["metadata"]}
        for c in chunks
        if c["type"] == "parent"
    }
    return search_units, parents


# ---------------------------------------------------------------------------
# 벡터 색인 (Qdrant, 로컬 파일 임베디드 모드)
# ---------------------------------------------------------------------------
def _clean_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Qdrant payload는 JSON 직렬화 가능한 값만 허용. None은 빈 문자열로 치환."""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        out[k] = "" if v is None else v
    return out


def build_vector_index(
    config: dict[str, Any],
    search_units: list[dict[str, Any]],
) -> None:
    """검색 단위를 임베딩해 Qdrant(로컬 파일 임베디드 모드)에 저장한다."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    persist_dir = config["paths"]["vectorstore_dir"]
    collection_name = config["indexing"]["collection_name"]
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    docs = [u["text"] for u in search_units]
    embeddings = encode_documents(config, docs)

    client = QdrantClient(path=persist_dir)
    # 재빌드 시 기존 컬렉션 초기화(임베딩 모델·청킹이 바뀌면 다시 만들어야 함)
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name,
        vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE),
    )

    points = []
    for idx, (u, emb) in enumerate(zip(search_units, embeddings)):
        m = _clean_meta(u["metadata"])
        m["chunk_type"] = u["type"]
        m["parent_id"] = u.get("parent_id", u["id"])  # article은 자기 자신이 parent
        m["doc_id"] = u["id"]
        points.append(PointStruct(id=idx, vector=emb, payload=m))
    client.upsert(collection_name=collection_name, points=points)
    client.close()  # 로컬 모드는 디스크 락을 잡으므로 명시적으로 해제


# ---------------------------------------------------------------------------
# BM25 통계
# ---------------------------------------------------------------------------
def build_bm25(config: dict[str, Any], search_units: list[dict[str, Any]]) -> None:
    """검색 단위를 한국어 토큰화해 BM25 코퍼스 통계를 계산·영속화한다."""
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize_ko(u["text"]) for u in search_units]
    bm25 = BM25Okapi(tokenized)

    payload = {
        "bm25": bm25,
        "ids": [u["id"] for u in search_units],
        "docs": [u["text"] for u in search_units],
        "metadatas": [
            {**_clean_meta(u["metadata"]), "parent_id": u.get("parent_id", u["id"])}
            for u in search_units
        ],
        "tokenizer": "kiwipiepy",  # 질문도 동일 토크나이저로 토큰화해야 함
    }
    bm25_path = Path(config["paths"]["bm25_path"])
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump(payload, f)


# ---------------------------------------------------------------------------
# parent lookup 저장
# ---------------------------------------------------------------------------
def save_parents(config: dict[str, Any], parents: dict[str, dict[str, Any]]) -> None:
    out = Path(config["paths"]["vectorstore_dir"]) / "parents.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(parents, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 엔트리
# ---------------------------------------------------------------------------
def run(config: dict[str, Any]) -> dict[str, int]:
    """인덱싱 실행: 청크 로드 -> 벡터 색인 + BM25 + parent lookup."""
    chunks = load_chunks(config["paths"]["chunks_path"])
    search_units, parents = split_search_and_parents(chunks)

    build_vector_index(config, search_units)
    build_bm25(config, search_units)
    save_parents(config, parents)

    return {
        "search_units": len(search_units),
        "parents": len(parents),
    }
