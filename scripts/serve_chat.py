"""챗봇 데모 서버 — RAG 전 과정을 한 페이지에서 시각화한다.

표준 라이브러리 http.server만 사용한다(외부 웹 프레임워크 미도입 — 온프레미스·의존성 최소화).

엔드포인트:
  GET  /                 → web/index.html (단일 페이지 UI)
  POST /api/retrieve     → 질문 → 임베딩·매칭·프롬프트까지의 추적 결과(JSON)
  POST /api/generate     → 프롬프트 → 로컬 LLM 답변을 토큰 단위로 스트리밍(텍스트)
  GET  /api/health       → 로컬 LLM(Ollama) 가용 여부

사용법:
    python scripts/serve_chat.py            # http://localhost:8000
    python scripts/serve_chat.py --port 8080 --config config/experiments/exp1_retrieval.yaml

선행 조건: scripts/build_index.py 로 인덱스가 빌드되어 있어야 한다.
답변 생성을 쓰려면 로컬 LLM(Ollama)이 구동 중이어야 한다(없어도 검색 과정은 모두 보임).
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import generation, trace
from src.config import load_config, set_seed

WEB_DIR = ROOT / "web"

# 전역 설정(서버 기동 시 1회 로드). 워밍업 비용이 큰 임베더는 첫 질문에서 로드된다.
CONFIG: dict = {}


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.0 → 응답 끝에서 연결을 닫으므로, content-length 없이 스트리밍 가능.
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):  # 콘솔 로그 간소화
        sys.stderr.write("  %s\n" % (fmt % args))

    # ── 유틸 ───────────────────────────────────────────────────────
    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        return json.loads(body.decode("utf-8")) if body else {}

    def _send_json(self, obj: dict, status: int = 200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ── 라우팅 ─────────────────────────────────────────────────────
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (WEB_DIR / "index.html").read_bytes()
            self._send_bytes(html, "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._send_json({"llm_available": generation.is_backend_available(CONFIG),
                             "model": CONFIG["generation"]["model"]})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            if self.path == "/api/retrieve":
                self._handle_retrieve()
            elif self.path == "/api/generate":
                self._handle_generate()
            else:
                self._send_json({"error": "not found"}, 404)
        except Exception as exc:  # 데모 서버 — 오류를 화면에 그대로 전달
            import traceback
            traceback.print_exc()
            try:
                self._send_json({"error": str(exc)}, 500)
            except Exception:
                pass

    # ── 핸들러 ─────────────────────────────────────────────────────
    def _handle_retrieve(self):
        body = self._read_json()
        question = (body.get("question") or "").strip()
        if not question:
            self._send_json({"error": "질문이 비어 있습니다."}, 400)
            return
        result = trace.trace(CONFIG, question)
        result["llm_available"] = generation.is_backend_available(CONFIG)
        result["llm_model"] = CONFIG["generation"]["model"]
        self._send_json(result)

    def _handle_generate(self):
        body = self._read_json()
        prompt = body.get("prompt") or ""
        if not prompt:
            self._send_json({"error": "prompt 가 비어 있습니다."}, 400)
            return
        if not generation.is_backend_available(CONFIG):
            self._send_json({"error": "로컬 LLM(Ollama)이 응답하지 않습니다. "
                                      "Ollama를 구동하고 모델을 받은 뒤 다시 시도하세요."}, 503)
            return

        # 스트리밍 응답: content-length 없이 토큰 조각을 흘려보낸다.
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for chunk in generation.generate_stream(CONFIG, prompt):
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
        except Exception as exc:
            try:
                self.wfile.write(f"\n[생성 오류] {exc}".encode("utf-8"))
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="RAG 챗봇 데모 서버")
    parser.add_argument("--config", default=None, help="실험 오버라이드 YAML 경로")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    global CONFIG
    CONFIG = load_config(args.config)
    set_seed(CONFIG["seed"])

    # 워밍업: 임베딩 모델·Qdrant 클라이언트·BM25 색인을 미리 로드한다.
    # 이렇게 하면 첫 질문의 "임베딩 시간"이 1회성 모델 로드가 아니라
    # 실제 질문당 임베딩 지연만 반영해 화면 수치가 정직해진다.
    print("워밍업 중(임베딩 모델·색인 로드)…", flush=True)
    try:
        trace.trace(CONFIG, "워밍업")
        print("워밍업 완료.", flush=True)
    except Exception as exc:
        print(f"[경고] 워밍업 실패(첫 질문이 느릴 수 있음): {exc}", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print("=" * 60)
    print("  RAG 챗봇 데모 서버 시작")
    print(f"  주소     : {url}")
    print(f"  임베딩   : {CONFIG['embedding']['model']}")
    print(f"  LLM      : {CONFIG['generation']['model']}  "
          f"({'응답 가능' if generation.is_backend_available(CONFIG) else 'Ollama 미응답 — 검색 과정만 표시'})")
    print("  종료     : Ctrl+C")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
        server.shutdown()


if __name__ == "__main__":
    main()
