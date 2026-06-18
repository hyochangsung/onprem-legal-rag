# 온프레미스 법조문 RAG 시스템 — 설계 문서

- 작성일: 2026-06-17
- 대상: 법조문 형식(편-장-절-조-항-호) 사내 규정집(약 100쪽) 온프레미스 RAG
- 상위 지침: `CLAUDE.md`, 조사 배경: `docs/RAG.md`

---

## 1. 목표와 제약

### 목표
구조화된 사내 규정집을 대상으로, 일상어 질문과 조항·용어 질문 양쪽에 정확히 답하는
온프레미스 RAG 시스템을 구축한다. 모든 설계 변수는 실험으로 검증·확정한다.

### 제약 (반드시 준수)
- **온프레미스 + 로컬 LLM**: 외부 임베딩/LLM API(OpenAI, Cohere, Anthropic 등) 호출 전면 금지.
- **GPU 자원 제한**: 모델 크기·배치 크기 선택 시 항상 메모리를 고려.
- **문서 규모**: 약 100쪽 → 청크 수백~수천 개. ANN 최적화 불필요(완전 탐색으로 충분).
- **원본 형태**: 구조화된 텍스트/MD(편-장-절-조 헤더 존재) → 규칙 기반 조 단위 파싱 가능.
- **재현성**: 랜덤 시드 고정. 실험 변수는 코드가 아닌 config로만 관리.
- **문서·주석**: 한국어.

---

## 2. 구성 방식 결정

**모듈형 파이썬 패키지 + config 구동 파이프라인**을 채택한다.

선택 이유(이 프로젝트가 실험 프로젝트이기 때문):
1. **변인 통제**: 실험 4종이 "한 단계(부품)만 교체"하는 구조라, 단계가 모듈로 분리돼야
   나머지를 고정한 채 한 변수만 바꿔 동일 질문셋으로 비교할 수 있다.
2. **진행 순서 대응**: `청킹 → 인덱싱 → 검색 → 평가` 순서가 모듈 경계와 1:1로 대응.
3. **단계별 독립 검증**: 정확도 저하 시 원인(청킹/임베딩/검색)을 단위 테스트로 좁힐 수 있다.
4. **config 분리의 전제**: 실험 변수를 config로 빼려면 그 값을 읽는 코드 경계(모듈)가 필요.

> 노트북 중심(B)은 재현성·변인 통제 원칙과 충돌하여 배제.
> 프레임워크 의존(C, LlamaIndex/Haystack)은 법조문 전용 파싱·Parent-Child·RRF 튜닝
> 커스터마이즈가 번거롭고 온프레미스 의존성 최소화 원칙과 어긋나 배제.
> 단, 부품 라이브러리(Qdrant, rank_bm25, sentence-transformers)는 그대로 활용한다.

"모듈형"의 범위: 마이크로서비스가 아니라 `src/`를 단계별 파일로 나누고 각 파일이
config를 입력받는 가벼운 수준. 100쪽 규모에 맞게 경량으로 구성.

---

## 3. 폴더 구조

```
llmagent-pjt/
├── CLAUDE.md
├── docs/
│   ├── RAG.md
│   └── superpowers/specs/          # 설계 문서 저장 위치
├── config/
│   ├── default.yaml                # 공통 기본값 (모든 실험 변수의 기준값)
│   └── experiments/                # 실험별 오버라이드만 담음
│       ├── exp1_retrieval.yaml     #  검색 방식 비교 (sparse/dense/hybrid, RRF 비율)
│       ├── exp2_chunking.yaml      #  조 단위 단일 vs Parent-Child
│       ├── exp3_reranker.yaml      #  re-ranker on/off
│       └── exp4_embedding.yaml     #  임베딩 모델 후보 비교
├── data/
│   ├── raw/                        # 원본 규정집 (구조화 MD/텍스트)
│   ├── processed/                  # 청킹 결과 (chunks.jsonl: 항=child, 조=parent)
│   └── eval/
│       └── qa_set.jsonl            # 질문-정답 50개 (유형 태그: 일상어 / 조항·용어)
├── src/
│   ├── config.py                   # YAML 로더 (default + 실험 오버라이드 병합), 시드 고정
│   ├── chunking.py                 # 편-장-절-조-항 파서, Parent-Child, 표 요약 부착
│   ├── indexing.py                 # 임베딩→Qdrant, BM25 통계, 가상질문 인덱싱, 메타데이터 저장
│   ├── retrieval.py                # BM25+벡터, RRF 결합, 메타데이터 필터, cross-encoder 리랭크
│   ├── generation.py               # 로컬 LLM 답변 생성 (조 전체 컨텍스트 주입)
│   └── evaluation.py               # Hit Rate@k, MRR, RAGAS
├── scripts/
│   ├── build_index.py              # 청킹+인덱싱 1회 실행
│   ├── run_query.py                # 질문 1개 → 검색→답변 (수동 확인용)
│   └── run_experiment.py           # config 받아 평가셋 전체 돌리고 결과표 출력
├── results/                        # 실험 결과 표 (csv/md), 실험마다 누적
├── tests/                          # 단계별 단위 테스트
├── vectorstore/                    # Qdrant 영속화 (gitignore)
├── requirements.txt
└── .gitignore
```

핵심 원칙: `src/`의 각 파일이 파이프라인 한 단계이며, 실험 변수는 코드에 없고
`config/`에만 존재한다. `run_experiment.py`가 config를 바꿔가며 같은
`qa_set.jsonl`으로 돌려 결과를 `results/`에 누적한다.

---

## 4. 파이프라인 구성요소 (단위와 책임)

### 4.1 chunking.py
- 입력: `data/raw/`의 구조화 MD/텍스트
- 동작: 편-장-절-조-항-호 파싱 → **항=child / 조=parent** 구조 생성.
  각 청크에 편-장-절-조 경로를 메타데이터로 부착. 표는 위에 요약 텍스트를 붙여 함께 저장.
  너무 긴 조는 필요 시 재귀적 분할로 보정.
- 출력: `data/processed/chunks.jsonl`
- config 변수: 청킹 방식(조 단위 단일 청크 vs Parent-Child) 토글 — 실험2.

### 4.2 indexing.py
> "Hybrid"는 검색 시점의 결합 방식이고, 인덱싱은 그 Hybrid가 쓸 색인 2개를 각각 준비한다.
- **벡터 색인**: 청크 임베딩(기본 Snowflake/snowflake-arctic-embed-l-v2.0) → Qdrant 저장.
- **BM25용 코퍼스 통계 사전 계산**: 한국어 토큰화 + 청크별 단어 빈도/IDF/길이 통계 계산 후
  영속화(pickle). (수백~수천 청크라 속도용 역색인이 목적이 아니라, 코퍼스 통계를 1회 계산해
  매 질문마다 재사용하기 위함.)
- **가상 질문 인덱싱**: child가 답이 되는 가상 질문을 로컬 LLM으로 생성해 별도 벡터로 추가
  (일상어 질문 ↔ 규정 원문 간극 해소).
- 메타데이터: 편-장-절-조 경로를 청크마다 저장(검색 시 범위 필터용).
- config 변수: 임베딩 모델(실험4), 가상질문 인덱싱 on/off.

### 4.3 retrieval.py (Hybrid 발생 지점)
- 질문 임베딩 → 벡터 검색(상위 k) + BM25 검색(상위 k)
- **RRF로 결합**: 점수가 아닌 순위를 사용해 `1/(k+rank)` 합산(스케일이 다른 두 점수를
  안전하게 통합). sparse/dense 가중치 비율은 config(실험1).
- (옵션) 메타데이터 필터로 범위 선제 축소.
- (옵션) cross-encoder 리랭커로 정밀 재정렬.
- 출력: 최종 조(parent) 컨텍스트.
- config 변수: top-k, RRF sparse/dense 비율(실험1), 리랭커 on/off(실험3).

### 4.4 generation.py
- 로컬 LLM에 검색된 조 전체를 컨텍스트로 주입해 답변 생성.
- 서빙 환경이 아직 없으므로 **Ollama 권장**(온프레미스·구동 단순). 모델·엔드포인트는 config화.
- config 변수: LLM 모델/엔드포인트, 프롬프트 템플릿.

### 4.5 evaluation.py
- 검색 채점: Hit Rate@k, MRR (정답 조 id 기준).
- 답변 채점: RAGAS 지표, 최종 답변 품질.
- 질문 유형별(일상어 / 조항·용어)로도 분리 집계.

---

## 5. 데이터 흐름

```
[원본 MD] --chunking--> [chunks.jsonl: child(항)/parent(조)]
                              |
                         indexing
                    /                \
            [Qdrant 벡터색인]   [BM25 통계 + 가상질문 벡터]
                    \                /
   [질문] --embed--> retrieval(벡터 + BM25 --RRF--> 필터 --리랭크)
                              |
                      [최종 조(parent) 컨텍스트]
                              |
                        generation(로컬 LLM)
                              |
                          [답변] --evaluation--> Hit Rate/MRR/RAGAS
```

---

## 6. config로 분리할 실험 변수 (하드코딩 금지)

- 임베딩 모델 (후보 2~3개)
- RRF sparse/dense 가중치 비율
- top-k
- Re-ranker 적용 여부
- 청킹 방식 (조 단위 단일 vs Parent-Child)
- 가상 질문 인덱싱 on/off
- ANN 적용 여부 (기본 off; 소규모라 완전 탐색)
- 랜덤 시드(고정)

`config/default.yaml`이 기준값을 갖고, `config/experiments/*.yaml`은 바뀌는 변수만
오버라이드한다.

---

## 7. 단계별 구현 계획 (진행 순서: 청킹 → 인덱싱 → 검색 → 평가)

| 단계 | 산출물 | 핵심 내용 |
| --- | --- | --- |
| 0. 스캐폴딩 & config 골격 | 폴더, requirements, config.py, default.yaml | config 병합 로더, 시드 고정, 변수 자리 정의 |
| 1. 청킹 | chunking.py, chunks.jsonl | 조-항 파서, Parent-Child, 표 요약, 메타데이터 |
| 2. 인덱싱 | indexing.py, vectorstore/, BM25 통계 | 임베딩→Qdrant, BM25 통계, 가상질문 인덱싱 |
| 3. 검색 | retrieval.py | 벡터+BM25, RRF 결합, 메타 필터, 리랭커 |
| 4. 로컬 LLM + 생성 | generation.py | Ollama 구축, 조 전체 주입 답변 생성 |
| 5. 평가셋 구축 | qa_set.jsonl (50개) | 질문-정답, 유형 태그, 근거 조 id |
| 6. 평가 | evaluation.py | Hit Rate@k, MRR, RAGAS, 결과표 누적 |
| 7. 실험 실행 | results/ 표 | 실험1~4를 동일 질문셋으로 한 변수씩 비교 |

### 진행 예정 실험
1. 검색 방식 비교: Sparse / Dense / Hybrid (RRF 비율 튜닝 포함), 질문 유형별.
2. 청킹 비교: 조 단위 단일 청크 / Parent-Child (Hit Rate·MRR·답변 품질).
3. Re-ranker 유무 비교: 정확도 향상분과 지연 증가를 함께 측정(교환비).
4. 임베딩 모델 비교: 정확도가 충분히 나오는 가장 작은 모델 탐색.

---

## 8. 평가 체계

- 지표: Hit Rate@k, MRR, 최종 답변 품질, RAGAS.
- 평가셋: 질문-정답 약 50개. 유형 구분(일상어 / 조항·용어).
- 원칙: 모든 실험은 동일 질문셋으로 비교. 결과는 표로 기록(`results/`).

---

## 9. 범위 밖 (후순위 — 현 단계 미구현)

- GraphRAG, Hierarchical Indexing(RAPTOR 등).
- ANN(HNSW 등) 최적화 — 소규모라 불필요.
- 외부 API 사용 코드 일체.

---

## 10. 기술 스택

- 언어: Python
- 벡터 DB: Qdrant (로컬 파일 임베디드 모드, 서버 불필요·소규모)
- Sparse: rank_bm25 + 한국어 토크나이저
- 임베딩/리랭커: sentence-transformers (cross-encoder)
- 로컬 LLM 서빙: Ollama (권장)
- 평가: RAGAS
- 설정: PyYAML
