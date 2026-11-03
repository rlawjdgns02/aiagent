"""내부 문서 검색과 장기 기억 도구. 임베딩 모델은 첫 사용 시 로드한다."""
from __future__ import annotations

import json
from typing import Any
import numpy as np
from config import (DOCUMENTS_FILE, MEMORY_FILE, EMBEDDING_MODEL,
                    MIN_DOCUMENT_SCORE, MIN_MEMORY_SCORE, MEMORY_DUPLICATE_SCORE)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search internal company documents when a plan step requires "
                "company-specific rules, procedures, or policies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A focused query for the current plan step. Preserve "
                            "the important constraints stated in that step."
                        ),
                    },
                    "status_filter": {
                        "type": "string",
                        "enum": ["active", "retired"],
                        "description": (
                            "Optional. Restrict results to documents with this "
                            "status. Use 'active' when the step asks about the "
                            "current policy."
                        ),
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": (
                "Save one durable user requirement, preference, or decision "
                "explicitly quoted in the current plan step."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Exactly one durable user-provided fact.",
                    }
                },
                "required": ["content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Search stored user requirements, preferences, decisions, or "
                "long-running task information required by the current plan step."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A concise description of the user information needed.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


with DOCUMENTS_FILE.open("r", encoding="utf-8") as document_file:
    DOCUMENTS = json.load(document_file)
_embedding_model: Any = None
_document_embeddings: np.ndarray | None = None


def get_embedding_model() -> Any:
    """검색이 실제로 필요해질 때 임베딩 의존성과 모델을 불러온다."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_document_embeddings() -> np.ndarray:
    global _document_embeddings
    if _document_embeddings is None:
        texts = [document["text"] for document in DOCUMENTS]
        _document_embeddings = get_embedding_model().encode(
            texts,
            normalize_embeddings=True,
        )
    return _document_embeddings


def search_documents(
    query: str,
    top_k: int = 3,
    status_filter: str | None = None,
) -> list[dict[str, Any]]:
    query_embedding = get_embedding_model().encode(
        [query],
        prompt_name="query",
        normalize_embeddings=True,
    )[0]
    scores = get_document_embeddings() @ query_embedding

    candidates = [
        {**document, "score": float(scores[index])}
        for index, document in enumerate(DOCUMENTS)
        if status_filter is None or document["status"] == status_filter
    ]
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return [
        candidate for candidate in candidates[:top_k]
        if candidate["score"] >= MIN_DOCUMENT_SCORE
    ]


def embed_passage(text: str) -> list[float]:
    return get_embedding_model().encode(
        text,
        normalize_embeddings=True,
    ).tolist()


def embed_query(text: str) -> list[float]:
    return get_embedding_model().encode(
        text,
        prompt_name="query",
        normalize_embeddings=True,
    ).tolist()


def load_memories() -> list[dict[str, Any]]:
    if not MEMORY_FILE.exists():
        return []
    with MEMORY_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


memories = load_memories()


def save_memories() -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MEMORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(memories, file, ensure_ascii=False, indent=2)


def reset_memory_store() -> None:
    global memories
    memories = []
    if MEMORY_FILE.exists():
        MEMORY_FILE.unlink()


def memory_write(content: str) -> dict[str, Any]:
    embedding = embed_passage(content)

    best_score = 0.0
    best_memory: dict[str, Any] | None = None
    for memory in memories:
        score = float(np.dot(embedding, np.asarray(memory["embedding"])))
        if score > best_score:
            best_score = score
            best_memory = memory

    if best_memory is not None and best_score >= MEMORY_DUPLICATE_SCORE:
        if content == best_memory["content"]:
            return {"status": "unchanged", "id": best_memory["id"]}
        previous_content = best_memory["content"]
        best_memory["content"] = content
        best_memory["embedding"] = embedding
        save_memories()
        return {
            "status": "updated",
            "id": best_memory["id"],
            "previous_content": previous_content,
        }

    next_id = max((memory["id"] for memory in memories), default=-1) + 1
    memories.append(
        {
            "id": next_id,
            "content": content,
            "embedding": embedding,
        }
    )
    save_memories()
    return {"status": "saved", "id": next_id}


def search_memory(query: str, k: int = 5) -> dict[str, Any]:
    query_embedding = embed_query(query)
    scored = [
        (float(np.dot(query_embedding, np.asarray(memory["embedding"]))), memory)
        for memory in memories
    ]
    scored.sort(key=lambda item: item[0], reverse=True)

    results = [
        {
            "id": memory["id"],
            "content": memory["content"],
            "score": round(score, 3),
        }
        for score, memory in scored[:k]
        if score >= MIN_MEMORY_SCORE
    ]
    return {"query": query, "results": results}


TOOL_FUNCTIONS = {
    "search_documents": search_documents,
    "search_memory": search_memory,
    "memory_write": memory_write,
}

