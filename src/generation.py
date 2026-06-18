"""생성 — 로컬 LLM(Ollama) 답변 생성.

온프레미스 제약: 외부 LLM API 금지. 로컬 Ollama 엔드포인트만 사용한다.
표준 라이브러리(urllib)만으로 Ollama HTTP API를 호출해 의존성을 최소화한다.

검색된 조 전체를 컨텍스트로 최종 답변을 생성하며, --judge 채점에서도 심판 LLM 호출에 쓰인다.

LLM이 아직 구동되지 않았을 수 있으므로, is_backend_available로 가용성을 먼저 확인한다.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def is_backend_available(config: dict[str, Any], timeout: float = 1.5) -> bool:
    """로컬 LLM 백엔드(Ollama)가 응답 가능한지 확인한다."""
    if config["generation"]["backend"] != "ollama":
        return False
    endpoint = config["generation"]["endpoint"].rstrip("/")
    try:
        with urllib.request.urlopen(f"{endpoint}/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def generate(config: dict[str, Any], prompt: str) -> str:
    """프롬프트를 로컬 LLM에 보내 생성 결과 텍스트를 반환한다."""
    gen = config["generation"]
    if gen["backend"] != "ollama":
        raise NotImplementedError(f"지원하지 않는 backend: {gen['backend']}")

    endpoint = gen["endpoint"].rstrip("/")
    payload = {
        "model": gen["model"],
        "prompt": prompt,
        "stream": False,
        "think": False,  # 사고 과정(thinking) 출력 비활성화 — 재현성·속도. 미지원 모델은 무시
        "options": {
            "temperature": gen["temperature"],
            "num_predict": gen["max_tokens"],
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint}/api/generate", data=data, headers={"Content-Type": "application/json"}
    )
    # 첫 추론은 모델 로드로 느릴 수 있어 넉넉히. config로 조정 가능.
    request_timeout = gen.get("request_timeout", 600)
    with urllib.request.urlopen(req, timeout=request_timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("response", "")


def generate_stream(config: dict[str, Any], prompt: str):
    """프롬프트를 로컬 LLM에 보내 생성 토큰을 스트리밍으로 yield한다.

    Ollama는 stream=True일 때 줄 단위 JSON(NDJSON)을 흘려보낸다. 각 줄의
    "response" 조각을 그대로 흘려보내 챗봇 화면에서 타이핑처럼 보이게 한다.
    """
    gen = config["generation"]
    if gen["backend"] != "ollama":
        raise NotImplementedError(f"지원하지 않는 backend: {gen['backend']}")

    endpoint = gen["endpoint"].rstrip("/")
    payload = {
        "model": gen["model"],
        "prompt": prompt,
        "stream": True,
        "think": False,
        "options": {
            "temperature": gen["temperature"],
            "num_predict": gen["max_tokens"],
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint}/api/generate", data=data, headers={"Content-Type": "application/json"}
    )
    request_timeout = gen.get("request_timeout", 600)
    with urllib.request.urlopen(req, timeout=request_timeout) as resp:
        for raw in resp:
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw.decode("utf-8"))
            chunk = obj.get("response", "")
            if chunk:
                yield chunk
            if obj.get("done"):
                break


def build_answer_prompt(question: str, context_articles: list[str]) -> str:
    """검색된 조 전체를 컨텍스트로 최종 답변 프롬프트를 구성한다."""
    context = "\n\n---\n\n".join(context_articles)
    return (
        "당신은 사내 규정집에 근거해 답변하는 어시스턴트입니다. "
        "아래 [참고 조항]에 있는 내용만 근거로, 한국어로 정확하게 답하세요. "
        "참고 조항에 근거가 없으면 모른다고 답하세요.\n\n"
        f"[참고 조항]\n{context}\n\n"
        f"[질문]\n{question}\n\n[답변]\n"
    )
