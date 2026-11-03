"""Tool schemas, retrieval, memory, and tool execution.

This module knows how tools work and how their results are stored. It does not
decide which plan step should run or whether the Agent should replan.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


MEMORY_FILE = Path(os.getenv("MEMORY_FILE", "memory_store.json"))

# Qwen3-Embedding-0.6B 기준 임계값
MIN_DOCUMENT_SCORE = 0.5
MIN_MEMORY_SCORE = 0.45
MEMORY_DUPLICATE_SCORE = 0.9


# Tool schemas
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Demonstration document collection
# ---------------------------------------------------------------------------

DOCUMENTS = [
    {
        "id": "travel_lodging_A",
        "text": (
            "Overseas travel lodging limit. The maximum lodging reimbursement "
            "is 180 dollars per night."
        ),
        "source": "travel_lodging_A.txt",
        "year": 2024,
        "status": "retired",
        "category": "travel",
    },
    {
        "id": "travel_lodging_B",
        "text": (
            "For international business trips, employees may claim accommodation "
            "expenses up to 220 dollars per night."
        ),
        "source": "travel_lodging_B.txt",
        "year": 2026,
        "status": "active",
        "category": "travel",
    },
    {
        "id": "travel_request",
        "text": (
            "International business travel requests must be approved by the "
            "department head before departure."
        ),
        "source": "travel_request.txt",
        "year": 2026,
        "status": "active",
        "category": "travel",
    },
    {
        "id": "usb_purchase",
        "text": (
            "USB storage device purchase requests must include a business "
            "justification and a valid cost center."
        ),
        "source": "usb_purchase.txt",
        "year": 2026,
        "status": "active",
        "category": "security",
    },
    {
        "id": "usb_security",
        "text": (
            "Personal removable storage devices, including USB flash drives, "
            "must not be connected to company computers."
        ),
        "source": "usb_security.txt",
        "year": 2026,
        "status": "active",
        "category": "security",
    },
    {
        "id": "meal_expense",
        "text": (
            "Meal expenses on approved business trips are reimbursed up to "
            "60 dollars per day."
        ),
        "source": "meal_expense.txt",
        "year": 2026,
        "status": "active",
        "category": "expense",
    },
]

_embedding_model: SentenceTransformer | None = None
_document_embeddings: np.ndarray | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
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


# ---------------------------------------------------------------------------
# Persistent memory store
# ---------------------------------------------------------------------------

# Qwen3-Embedding은 질의 쪽에만 instruction prompt를 붙인다.
# (모델 내장 프롬프트를 prompt_name="query"로 사용; passage는 원문 그대로)

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




# ---------------------------------------------------------------------------
# Tools: execute globally, save results with plan-step provenance
# ---------------------------------------------------------------------------

def execute_tool(
    name: str,
    args: dict[str, Any],
    state: dict[str, Any],
    step: dict[str, Any],
) -> dict[str, Any]:
    current_task = state["current_task"]
    plan_step_id = step["id"]

    if name == "search_documents":
        documents = search_documents(**args)
        for document in documents:
            document_id = document["id"]
            if document_id not in current_task["retrieved_documents"]:
                current_task["retrieved_documents"][document_id] = {
                    **document,
                    "retrieved_by_plan_steps": [],
                }
            provenance = current_task["retrieved_documents"][document_id][
                "retrieved_by_plan_steps"
            ]
            if plan_step_id not in provenance:
                provenance.append(plan_step_id)
            if document_id not in step["document_ids"]:
                step["document_ids"].append(document_id)

        if not documents:
            return {
                "status": "no_results",
                "note": (
                    "No documents scored above the relevance threshold. "
                    "Rephrase the query once, or state that the information "
                    "is not available in the internal documents."
                ),
            }

        return {
            "status": "saved",
            "state_key": "current_task.retrieved_documents",
            "document_ids": [document["id"] for document in documents],
        }

    if name == "search_memory":
        result = search_memory(**args)
        for memory in result["results"]:
            memory_id = str(memory["id"])
            if memory_id not in current_task["retrieved_memories"]:
                current_task["retrieved_memories"][memory_id] = {
                    **memory,
                    "retrieved_by_plan_steps": [],
                }
            provenance = current_task["retrieved_memories"][memory_id][
                "retrieved_by_plan_steps"
            ]
            if plan_step_id not in provenance:
                provenance.append(plan_step_id)
            if memory_id not in step["memory_ids"]:
                step["memory_ids"].append(memory_id)

        return {
            "status": "saved",
            "state_key": "current_task.retrieved_memories",
            "memory_ids": [memory["id"] for memory in result["results"]],
        }

    if name == "memory_write":
        return memory_write(**args)

    raise ValueError(f"Unknown tool: {name}")



