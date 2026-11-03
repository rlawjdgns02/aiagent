"""프로젝트 경로, 환경 변수, 실행·검색 설정을 한곳에서 관리한다."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env", override=False)


def env_bool(name: str, default: bool) -> bool:
    """환경 변수의 true/false를 명시적으로 해석한다."""
    value = os.getenv(name)
    if value is None:
        return default
    if value.strip().lower() in {"true", "1", "yes", "on"}:
        return True
    if value.strip().lower() in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false.")


def project_path(name: str, default: str) -> Path:
    """상대 경로는 프로젝트 폴더를 기준으로 해석한다."""
    path = Path(os.getenv(name, default)).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://agent.kau.ac.kr/v1")
API_KEY = os.getenv("KAU_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
MODEL_NAME = os.getenv("MODEL_NAME", "").strip()
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))
ENABLE_THINKING = env_bool("ENABLE_THINKING", False)
DEBUG = env_bool("DEBUG", False)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
DOCUMENTS_FILE = project_path("DOCUMENTS_FILE", "data/documents.json")
MEMORY_FILE = project_path("MEMORY_FILE", "data/memory_store.json")
OUTPUT_DIR = project_path("OUTPUT_DIR", "outputs")

# 원본 Agent와 같은 실행·검색 기준을 사용한다.
RECENT_TURNS = 2
USE_SUMMARY = True
MAX_PLAN_STEPS = 8
MAX_REPLANS = 2
MAX_AGENT_LOOP_ITERATIONS = 6
MAX_TOTAL_LLM_CALLS = 30
MAX_JSON_ATTEMPTS = 3
MIN_DOCUMENT_SCORE = 0.5
MIN_MEMORY_SCORE = 0.45
MEMORY_DUPLICATE_SCORE = 0.9

