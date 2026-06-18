# 온프레미스 법조문 RAG 시스템

법조문 형식(편-장-절-조-항-호)의 사내 규정집을 대상으로 하는 **온프레미스 RAG** 시스템입니다.
외부 API 없이 로컬 임베딩·로컬 LLM만으로 동작하며, 모든 설계 변수는 실험으로 검증·확정하는 것을 목표로 합니다.

- 상위 지침: [`CLAUDE.md`](CLAUDE.md)
- 조사 배경: [`docs/RAG.md`](docs/RAG.md)
- 설계 문서: [`docs/superpowers/specs/2026-06-17-onprem-legal-rag-design.md`](docs/superpowers/specs/2026-06-17-onprem-legal-rag-design.md)

---

## 1. 핵심 설계

| 항목 | 선택 |
| --- | --- |
| 청킹 | 구조 기반(조 단위) + Parent-Child. **항(child)으로 검색, 조(parent) 전체를 LLM에 전달** |
| 임베딩 | 한국어 로컬 모델 (`Snowflake/snowflake-arctic-embed-l-v2.0`, 최종은 실험으로 확정) |
| 검색 | Hybrid (BM25 + 벡터), **RRF**로 결합 + (옵션) cross-encoder 리랭커 |
| 메타데이터 | 편-장-절-조 경로를 청크마다 저장 → 범위 선제 축소 |
| 생성 | 로컬 LLM (Ollama), 검색된 조 전체를 컨텍스트로 주입 |
| 평가 | Hit Rate@k, MRR, (옵션) 로컬 LLM 답변 채점 |

**제약**: 외부 임베딩/LLM API 전면 금지, GPU 자원 제한 고려, 약 100쪽 규정집(ANN 불필요).

---

## 2. 폴더 구조

```
llmagent-pjt/
├── config/
│   ├── default.yaml                # 모든 실험 변수의 기준값
│   └── experiments/                # 실험별 오버라이드만 담음 (exp1~exp4, remote_colab)
├── data/
│   ├── raw/regulations.md          # 원본 규정집 (구조화 MD)
│   ├── processed/chunks.jsonl      # 청킹 산출물 (gitignore)
│   └── eval/qa_set.jsonl           # 질문-정답 평가셋 (유형 태그·정답 조 id)
├── src/
│   ├── config.py                   # YAML 2층 병합 로더 + 시드 고정
│   ├── chunking.py                 # 편-장-절-조-항 파서, Parent-Child, 표 요약
│   ├── indexing.py                 # 임베딩→Qdrant, BM25 통계
│   ├── retrieval.py                # BM25+벡터, RRF, 메타필터, 리랭커
│   ├── generation.py               # 로컬 LLM(Ollama) 답변 생성
│   ├── evaluation.py               # Hit Rate@k, MRR, 답변 채점
│   └── trace.py                    # 챗봇 데모용 — 검색 전 과정(임베딩·매칭·프롬프트) 추적
├── scripts/
│   ├── build_index.py              # 청킹 + 인덱싱 1회 실행
│   ├── run_query.py                # 질문 1개 → 검색 → 답변 (수동 확인용)
│   ├── run_experiment.py           # 평가셋 전체 실행 + 결과표 누적
│   └── serve_chat.py               # 챗봇 데모 서버 (검색 과정 시각화, §4.4)
├── web/index.html                  # serve_chat.py가 서빙하는 단일 페이지 UI
├── tests/                          # 단계별 단위 테스트
├── results/                        # 실험 결과표 (gitignore되지 않음, 누적)
├── vectorstore/                    # Qdrant 영속화 (로컬 파일 임베디드 모드, gitignore)
└── requirements.txt
```

---

## 3. 설치

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

로컬 LLM(답변 생성·답변 채점에 필요)은 [Ollama](https://ollama.com)를 사용합니다.

```bash
ollama serve                       # 백엔드 구동
ollama pull qwen2.5:7b-instruct    # config/default.yaml의 generation.model과 일치시킬 것
```

> 검색 채점(Hit Rate@k·MRR)만 할 경우 LLM 없이도 동작합니다.

**RAGAS 설치 확인 필요.** `requirements.txt`에 포함돼 있으나 의존성이 무거워(`datasets`·`langchain`·`openai` 등 동반) 환경에 따라 `pip install -r requirements.txt`에서 누락될 수 있습니다. 설치 여부를 확인하고, 안 돼 있으면 별도로 설치하세요.

```bash
python -c "import ragas; print(ragas.__version__)"   # 미설치면 ModuleNotFoundError
pip install "ragas>=0.2"                              # 누락 시 설치
```

> RAGAS는 `openai`·`langchain_openai`를 끌고 들어오며 기본 심판이 외부 API입니다. 온프레미스로 쓰려면 반드시 로컬 LLM·로컬 임베딩으로 배선해야 합니다(§8 참조).

---

## 4. 사용법

진행 순서는 **청킹 → 인덱싱 → 검색 → 평가** 입니다.

### 4.1 인덱스 빌드 (청킹 + 인덱싱)

```bash
python scripts/build_index.py
# 인덱스에 영향을 주는 변수(청킹 방식·임베딩 모델)를 바꾼 실험은 --config로 재빌드
python scripts/build_index.py --config config/experiments/exp4_embedding.yaml
```

### 4.2 질문 1개 실행 (수동 확인)

```bash
python scripts/run_query.py "전결 금액 기준이 어떻게 되나요"
python scripts/run_query.py "..." --no-generate         # 검색 결과만, LLM 생략
```

### 4.3 평가셋 전체 실행 (실험)

```bash
python scripts/run_experiment.py --label baseline                       # default 설정
python scripts/run_experiment.py --config config/experiments/exp1_retrieval.yaml --label exp1-sparse
python scripts/run_experiment.py --label baseline --judge               # 답변 품질까지 채점(LLM 필요)
```

결과는 `results/summary.md`·`summary.csv`에 누적되고, 질문별 상세는 `results/<label>_per_question.csv`에 저장됩니다.

### 4.4 챗봇 데모 서버 (검색 과정 시각화)

```bash
python scripts/serve_chat.py                     # 로컬 설정으로 http://localhost:8000
python scripts/serve_chat.py --port 8080 --config config/experiments/exp1_retrieval.yaml
```

로컬 GPU(T600 4GB 등)가 약해 생성이 느릴 때는, Colab의 외부 GPU에서 Ollama를 띄우고
생성만 그쪽으로 위임하는 오버라이드를 쓸 수 있습니다.

```bash
python scripts/serve_chat.py --config config/experiments/remote_colab.yaml
```

> ⚠️ **주의**: 이 오버라이드는 검색된 규정 조문이 프롬프트에 담겨 외부(Google Colab)로 전송되므로
> `CLAUDE.md` 1번의 "온프레미스 + 외부 API 금지" 원칙과 충돌합니다. **연구·실험 단계에서만** 사용하고
> 실서비스에는 로컬 설정으로 복귀해야 합니다. 사용 전 `config/experiments/remote_colab.yaml`의
> `colab/colab_ollama_server.ipynb` 터널 URL(`endpoint`)이 매 세션 갱신되었는지 확인하세요.

선행 조건은 4.1의 인덱스 빌드와 동일합니다. 답변 생성을 쓰려면 로컬이든 Colab이든 Ollama가 응답 가능해야 하며,
없어도 검색 과정(`/api/retrieve`)은 화면에 그대로 표시됩니다.

---

## 5. 설정 (config)

실험으로 정할 값은 코드에 하드코딩하지 않고 전부 `config/`에서 관리합니다.

- `config/default.yaml` — 모든 변수의 기준값(baseline)
- `config/experiments/*.yaml` — 이번 실험에서 **바뀌는 변수만** 오버라이드 (default 위에 깊은 병합)

주요 실험 변수: 임베딩 모델, RRF sparse/dense 가중치, top-k, 리랭커 on/off·모델, 청킹 방식(조 단위 단일 vs Parent-Child).

---

## 6. 평가

- **검색 채점**: 정답 조(jo) id 기준 Hit Rate@k, MRR. 전체 + 질문 유형별(일상어 / 조항용어) 집계.
- **답변 채점**: 온프레미스 제약상 외부 API 금지 → 로컬 LLM을 심판으로 쓰는 경량 채점(`--judge`). 백엔드 없으면 자동 스킵.
- **원칙**: 모든 실험은 동일 평가셋(`data/eval/qa_set.jsonl`)으로 비교.

진행 예정 실험: ① 검색 방식 비교(Sparse/Dense/Hybrid·RRF 비율) ② 청킹 비교 ③ 리랭커 유무(정확도 vs 지연) ④ 임베딩 모델 비교.

---

## 7. 테스트

```bash
python tests/test_chunking.py
python tests/test_retrieval.py
python tests/test_evaluation.py
# (인덱싱·생성 테스트는 모델/백엔드가 필요할 수 있음)
```

---

## 8. 남은 작업 (TODO)

### 데이터·환경 (선행 조건)
- [x] **전체 규정집 확정** — `data/raw/regulations.md`를 100조 전체본으로 교체 완료.
- [x] **평가셋 확장·정제** — 24문항(14개 조)에서 76문항(편2~5의 26개 절을 절당 2문항씩 커버)으로 확장. 100개 조 중 66개 조를 정답 근거로 직접 다룸.

### 파이프라인 완성
- [ ] **인덱스 최초 빌드** — `data/processed/`, `vectorstore/`가 아직 비어 있어 한 번도 빌드되지 않음. `build_index.py` 실행 + 100조 전체에서 청킹 파서가 정상 동작하는지 확인 필요.
- [ ] **리랭커 모델 지정** — `config`의 `retrieval.reranker.model`이 비어 있어 현재 재정렬 스킵. 한국어 cross-encoder 모델 선정 후 지정(실험3).
- [ ] **RAGAS 정식 연동** — 설치 완료(ragas 0.4.3). 다만 기본 심판·임베딩이 외부 API(OpenAI)라 온프레미스에선 그대로 못 씀. 로컬 LLM·로컬 임베딩으로 배선해야 사용 가능. 그 전까지 답변 품질은 로컬 LLM 심판 채점(`--judge`)으로 대체.

### 실험 (7단계)
- [ ] **실험 1** 검색 방식 비교 (Sparse / Dense / Hybrid, RRF 비율 튜닝), 질문 유형별.
- [ ] **실험 2** 청킹 비교 (조 단위 단일 청크 / Parent-Child).
- [ ] **실험 3** 리랭커 유무 비교 (정확도 향상분 vs 지연 증가 교환비).
- [ ] **실험 4** 임베딩 모델 비교 (정확도 충분한 가장 작은 모델 탐색).

### 후순위 (현 단계 미구현)
- GraphRAG, Hierarchical Indexing(RAPTOR 등), ANN(HNSW) 최적화 — 소규모라 불필요/후순위.
