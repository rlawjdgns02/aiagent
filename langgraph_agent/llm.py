"""OpenAI 호환 호출, JSON 검증·재시도, 실행 예산과 SDK 메시지 변환."""
from __future__ import annotations

import json
import re
from typing import Any, Callable
from agent_state import TaskState
from config import (OPENAI_BASE_URL, API_KEY, REQUEST_TIMEOUT, ENABLE_THINKING,
                    MAX_TOTAL_LLM_CALLS, MAX_JSON_ATTEMPTS)


def create_client() -> Any:
    """실제 실행 시에만 OpenAI SDK 클라이언트를 만든다."""
    from openai import OpenAI
    return OpenAI(base_url=OPENAI_BASE_URL, api_key=API_KEY, timeout=REQUEST_TIMEOUT)


def call_completion(client: Any, **request: Any) -> Any:
    """Qwen/vLLM의 thinking 설정을 모든 단계에 동일하게 적용한다."""
    request.setdefault("extra_body", {
        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}
    })
    return client.chat.completions.create(**request)


class TurnBudgetExceeded(RuntimeError):
    """현재 turn의 LLM 호출 예산을 소진했다."""


def show_context(
    label: str,
    context: list[dict[str, Any]],
    debug: bool,
) -> None:
    if debug:
        print(f"\n===== {label} CONTEXT =====")
        for message in context:
            print(message["role"], ":", message.get("content", ""))


def count_llm_call(task_state: TaskState) -> None:
    """요청 전에 예산을 검사하고 실제 LLM 요청 시도를 센다."""
    if task_state["llm_call_count"] >= MAX_TOTAL_LLM_CALLS:
        raise TurnBudgetExceeded("The turn exceeded MAX_TOTAL_LLM_CALLS.")
    task_state["llm_call_count"] += 1


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
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    task_state: TaskState,
    validate: Callable[[dict[str, Any]], Any] | None = None,
) -> Any:
    """JSON 파싱·검증 실패 시 오류를 모델에 알려 수정 출력을 요청한다.

    기존 함수의 state['task'] 대신 복사된 TaskState를 직접 전달받는다.
    API 통신 오류는 호출자에게 전파하여 실행 실패로 확인할 수 있게 한다.
    """
    conversation = list(messages)
    last_error: Exception | None = None

    for _ in range(MAX_JSON_ATTEMPTS):
        count_llm_call(task_state)
        response = call_completion(
            client=client,
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
    """기존 OpenAI 형식의 dict를 유지하며 SDK 객체를 State에 넣지 않는다."""
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

