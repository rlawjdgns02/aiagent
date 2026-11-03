"""LLM 호출 헬퍼: 호출 예산, JSON 호출·재시도, 메시지 변환, 디버그 출력.

LLM 호출 수는 노드 안의 LLMCallBudget으로 세고,
노드가 끝날 때 budget.used를 llm_call_count로 반환한다.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from openai import OpenAI


MAX_TOTAL_LLM_CALLS = 30
MAX_JSON_ATTEMPTS = 3


class TurnBudgetExceeded(RuntimeError):
    """이 턴에 허용된 LLM 호출 예산을 다 썼다.

    노드가 잡아서 aborted_reason을 기록하고, 라우터가 final_answer로 보낸다.
    """


class LLMCallBudget:
    """노드 안에서 LLM 호출 수를 센다. 시작값은 State의 llm_call_count다."""

    def __init__(self, used: int) -> None:
        self.used = used

    def count(self) -> None:
        self.used += 1
        if self.used > MAX_TOTAL_LLM_CALLS:
            raise TurnBudgetExceeded("The turn exceeded MAX_TOTAL_LLM_CALLS.")


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("The model must return one JSON object.")
    return value


def call_json(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    budget: LLMCallBudget,
    validate: Callable[[dict[str, Any]], Any] | None = None,
) -> Any:
    conversation = list(messages)
    last_error: Exception | None = None

    for _ in range(MAX_JSON_ATTEMPTS):
        budget.count()
        response = client.chat.completions.create(
            model=model,
            messages=conversation,
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        try:
            output = parse_json_object(content)
            return validate(output) if validate else output
        except (json.JSONDecodeError, ValueError) as error:
            last_error = error
            conversation = conversation + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        f"Your previous output was invalid: {error} "
                        "Return corrected JSON only, following the required schema."
                    ),
                },
            ]

    raise ValueError(
        f"Invalid JSON output after {MAX_JSON_ATTEMPTS} attempts: {last_error}"
    )


def assistant_message_to_dict(message: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
    return result


def show_context(
    label: str,
    context: list[dict[str, Any]],
    debug: bool,
) -> None:
    if not debug:
        return
    print(f"\n===== {label} CONTEXT =====")
    for message in context:
        print(message["role"], ":", message.get("content", ""))
