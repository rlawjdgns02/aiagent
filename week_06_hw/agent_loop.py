"""Week 6: Plan–Execute–Replan Agent orchestration in raw Python.

Terminology
-----------
- turn: one user message through the final assistant response
- plan step: one explicit subtask in a plan
- agent loop iteration: one LLM decision inside a plan-step executor
- tool call: one function invocation selected by the LLM

Module boundary
---------------
- tools.py: tool definitions and execution
- contexts.py: phase-specific context projection
- agent_loop.py: state, planning, execution, replanning, and turn lifecycle
"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from typing import Any, Callable

from openai import OpenAI

from contexts import (
    MAX_PLAN_STEPS,
    build_executor_context,
    build_final_context,
    build_planner_context,
    build_progress_context,
    build_replanner_context,
    build_summary_context,
)
from tools import TOOLS, execute_tool


RECENT_TURNS = 2
MAX_REPLANS = 2
MAX_AGENT_LOOP_ITERATIONS = 6
MAX_TOTAL_LLM_CALLS = 30
MAX_JSON_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def create_task_state(goal: str) -> dict[str, Any]:
    return {
        "goal": goal,
        "plan": [],
        "current_plan_step_id": None,
        "plan_revision": 0,
        "replan_count": 0,

        # Full task-level evidence is retained here with plan-step provenance.
        "retrieved_documents": {},
        "retrieved_memories": {},

        "llm_call_count": 0,
        "aborted_reason": None,
        "trace": [],
    }


def create_state() -> dict[str, Any]:
    return {
        # Conversation state: survives across completed turns.
        "turns": [],
        "summary": "",

        # Current task state: replaced when the next turn starts.
        "current_task": None,
    }


def create_plan_step(
    plan_step_id: int,
    instruction: str,
    depends_on: list[int],
) -> dict[str, Any]:
    return {
        "id": plan_step_id,
        "instruction": instruction,
        "depends_on": depends_on,
        "status": "pending",
        "result": None,

        # Internal tool-call protocol messages for this plan step only.
        "messages": [
            {
                "role": "user",
                "content": "Execute the current plan step.",
            }
        ],

        # IDs of artifacts produced by this plan step.
        "document_ids": [],
        "memory_ids": [],
    }


def add_trace(state: dict[str, Any], event: str, **details: Any) -> None:
    state["current_task"]["trace"].append({"event": event, **details})


def get_plan_step(
    state: dict[str, Any],
    plan_step_id: int,
) -> dict[str, Any] | None:
    for step in state["current_task"]["plan"]:
        if step["id"] == plan_step_id:
            return step
    return None


def get_current_plan_step(state: dict[str, Any]) -> dict[str, Any] | None:
    plan_step_id = state["current_task"]["current_plan_step_id"]
    if plan_step_id is None:
        return None
    return get_plan_step(state, plan_step_id)


def next_pending_plan_step(state: dict[str, Any]) -> dict[str, Any] | None:
    for step in state["current_task"]["plan"]:
        if step["status"] == "pending":
            return step
    return None


def compact_plan(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": step["id"],
            "instruction": step["instruction"],
            "depends_on": step["depends_on"],
            "status": step["status"],
        }
        for step in state["current_task"]["plan"]
    ]




# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

class TurnBudgetExceeded(RuntimeError):
    """이 턴에 허용된 LLM 호출 예산을 다 썼다.

    run_turn이 잡아서 크래시 대신 '지금까지의 증거로 답하는' 경로로 강등한다.
    """


def count_llm_call(state: dict[str, Any]) -> None:
    state["current_task"]["llm_call_count"] += 1
    if state["current_task"]["llm_call_count"] > MAX_TOTAL_LLM_CALLS:
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
    state: dict[str, Any],
    validate: Callable[[dict[str, Any]], Any] | None = None,
) -> Any:
    conversation = list(messages)
    last_error: Exception | None = None

    for _ in range(MAX_JSON_ATTEMPTS):
        count_llm_call(state)
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


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------

def normalize_initial_plan(raw_steps: Any) -> list[dict[str, Any]]:
    
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Planner output must contain a non-empty 'steps' list.")

    if len(raw_steps) > MAX_PLAN_STEPS:
        raise ValueError(
            f"Planner returned {len(raw_steps)} steps but at most "
            f"{MAX_PLAN_STEPS} are allowed. Merge related subtasks into "
            "fewer, self-contained steps."
        )
        
    normalized = []
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            raise ValueError("Each plan step must be a JSON object.")

        instruction = str(raw_step.get("instruction", "")).strip()
        depends_on = raw_step.get("depends_on", [])
        if not instruction:
            raise ValueError(f"Plan step {index} has no instruction.")
        if not isinstance(depends_on, list) or not all(
            isinstance(item, int) and 1 <= item < index for item in depends_on
        ):
            raise ValueError(f"Plan step {index} has invalid dependencies.")

        normalized.append(
            create_plan_step(index, instruction, sorted(set(depends_on)))
        )
        
    return normalized


def normalize_replanned_steps(
    raw_steps: Any,
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Replanner must return non-empty 'remaining_steps'.")

    if len(raw_steps) > MAX_PLAN_STEPS:
        raise ValueError(
            f"Replanner returned {len(raw_steps)} steps but at most "
            f"{MAX_PLAN_STEPS} are allowed. Merge related subtasks into "
            "fewer, self-contained steps."
        )

    if not all(isinstance(raw_step, dict) for raw_step in raw_steps):
        raise ValueError("Each replanned step must be a JSON object.")

    plan = state["current_task"]["plan"]
    completed_ids = {step["id"] for step in plan if step["status"] == "completed"}
    first_id = max((step["id"] for step in plan), default=0) + 1

    # Model-chosen IDs are advisory: map them to the IDs we actually assign.
    id_map = {
        raw_step["id"]: first_id + offset
        for offset, raw_step in enumerate(raw_steps)
        if isinstance(raw_step.get("id"), int) and raw_step["id"] not in completed_ids
    }

    normalized = []
    for step_id, raw_step in enumerate(raw_steps, start=first_id):
        instruction = str(raw_step.get("instruction", "")).strip()
        if not instruction:
            raise ValueError(f"Replanned step {step_id} has no instruction.")

        dependencies = resolve_replanned_dependencies(
            raw_step.get("depends_on", []), step_id, completed_ids, id_map
        )
        normalized.append(create_plan_step(step_id, instruction, dependencies))
    return normalized


def resolve_replanned_dependencies(
    depends_on: Any,
    step_id: int,
    completed_ids: set[int],
    id_map: dict[int, int],
) -> list[int]:
    if not isinstance(depends_on, list):
        raise ValueError(f"Replanned step {step_id} has invalid dependencies.")

    resolved = set()
    for dependency in depends_on:
        if isinstance(dependency, int) and dependency in completed_ids:
            # Reference to an already completed step result.
            resolved.add(dependency)
        elif (
            isinstance(dependency, int)
            and dependency in id_map
            and id_map[dependency] < step_id
        ):
            # Reference to an earlier step in this new plan.
            resolved.add(id_map[dependency])
        else:
            raise ValueError(
                f"Replanned step {step_id} has invalid dependency "
                f"{dependency!r}. Dependencies must be completed step IDs "
                "or IDs of earlier new steps."
            )
    return sorted(resolved)




# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def create_plan(
    client: OpenAI,
    model: str,
    state: dict[str, Any],
    current_turn: list[dict[str, Any]],
    recent_turns: int | None,
    use_summary: bool,
    debug: bool,
) -> None:
    context = build_planner_context(
        state,
        current_turn,
        recent_turns,
        use_summary,
    )
    show_context("PLANNER", context, debug)
    state["current_task"]["plan"] = call_json(
        client, model, context, state,
        validate=lambda output: normalize_initial_plan(output.get("steps"))
    )
    state["current_task"]["current_plan_step_id"] = state["current_task"]["plan"][0]["id"]
    add_trace(state, "plan_created", plan=compact_plan(state))


# ---------------------------------------------------------------------------
# Plan-step Executor: isolated tool-call agent loop
# ---------------------------------------------------------------------------

def execute_plan_step(
    client: OpenAI,
    model: str,
    state: dict[str, Any],
    debug: bool,
) -> dict[str, str]:
    step = get_current_plan_step(state)
    if step is None:
        return {"status": "failed", "content": "No current plan step exists."}

    step["status"] = "in_progress"
    add_trace(
        state,
        "plan_step_started",
        plan_step_id=step["id"],
        instruction=step["instruction"],
        depends_on=step["depends_on"],
    )

    for iteration in range(1, MAX_AGENT_LOOP_ITERATIONS + 1):
        context = build_executor_context(state, step)
        show_context(
            f"EXECUTOR / PLAN STEP {step['id']} / ITERATION {iteration}",
            context,
            debug,
        )

        response = client.chat.completions.create(
            model=model,
            messages=context,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        assistant_message = response.choices[0].message
        step["messages"].append(assistant_message_to_dict(assistant_message))

        add_trace(
            state,
            "agent_loop_iteration",
            plan_step_id=step["id"],
            iteration=iteration,
            tool_call_count=len(assistant_message.tool_calls or []),
        )

        if not assistant_message.tool_calls:
            return {
                "status": "completed",
                "content": (assistant_message.content or "").strip(),
            }

        for tool_call in assistant_message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("Tool arguments must be a JSON object.")
                result = execute_tool(name, args, state, step)
                add_trace(
                    state,
                    "tool_called",
                    plan_step_id=step["id"],
                    tool=name,
                    arguments=args,
                    result_summary=result,
                )
            except Exception as error:
                result = {"status": "error", "message": str(error)}
                add_trace(
                    state,
                    "tool_error",
                    plan_step_id=step["id"],
                    tool=name,
                    message=str(error),
                )

            step["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    return {
        "status": "failed",
        "content": (
            "The plan-step executor exceeded "
            f"{MAX_AGENT_LOOP_ITERATIONS} agent loop iterations."
        ),
    }


# ---------------------------------------------------------------------------
# Progress Checker and Replanner
# ---------------------------------------------------------------------------
def validate_progress_output(output: dict[str, Any]) -> dict[str, str]:
    decision = output.get("decision")
    if decision not in {"continue", "replan", "finish"}:
        raise ValueError(f"Invalid progress decision: {decision}")
    reason = str(output.get("reason", "")).strip()
    if decision == "replan" and not reason:
        raise ValueError("Replan decision must include a reason.")
    return {"decision": decision, "reason": reason}

def check_progress(
    client: OpenAI,
    model: str,
    state: dict[str, Any],
    step_result: dict[str, str],
    debug: bool,
) -> dict[str, str]:
    step = get_current_plan_step(state)
    if step is None:
        raise RuntimeError("Progress check requires a current plan step.")

    context = build_progress_context(state, step, step_result)
    show_context("PROGRESS CHECKER", context, debug)
    
    result = call_json(client, model, context, state, validate=validate_progress_output)
    
    add_trace(
        state,
        "progress_checked",
        plan_step_id=step["id"],
        **result,
    )
    return result


def replan(
    client: OpenAI,
    model: str,
    state: dict[str, Any],
    reason: str,
    debug: bool,
) -> None:
    current_task = state["current_task"]
    if current_task["replan_count"] >= MAX_REPLANS:
        raise RuntimeError("The turn exceeded MAX_REPLANS.")

    failed_step = get_current_plan_step(state)
    if failed_step is None:
        raise RuntimeError("Replanning requires a current plan step.")

    context = build_replanner_context(state, failed_step, reason)
    show_context("REPLANNER", context, debug)
    new_steps = call_json(
        client, model, context, state, 
        validate=lambda output: normalize_replanned_steps(
            output.get("remaining_steps"), state
            )
        )

    failed_step["status"] = "replaced"
    for step in current_task["plan"]:
        if step["status"] == "pending":
            step["status"] = "replaced"

    current_task["plan"].extend(new_steps)
    current_task["current_plan_step_id"] = new_steps[0]["id"]
    current_task["plan_revision"] += 1
    current_task["replan_count"] += 1
    add_trace(
        state,
        "plan_revised",
        reason=reason,
        new_steps=[
            {
                "id": step["id"],
                "instruction": step["instruction"],
                "depends_on": step["depends_on"],
            }
            for step in new_steps
        ],
    )


# ---------------------------------------------------------------------------
# Final Answer and turn lifecycle
# ---------------------------------------------------------------------------

def generate_final_answer(
    client: OpenAI,
    model: str,
    state: dict[str, Any],
    current_turn: list[dict[str, Any]],
    recent_turns: int | None,
    use_summary: bool,
    debug: bool,
) -> str:
    context = build_final_context(
        state,
        current_turn,
        recent_turns,
        use_summary,
    )
    show_context("FINAL ANSWER", context, debug)

    # do not count this LLM call in the task's llm_call_count, as it is not part of the plan execution
    response = client.chat.completions.create(
        model=model,
        messages=context,
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def update_summary(
    client: OpenAI,
    model: str,
    state: dict[str, Any],
    recent_turns: int,
) -> None:
    context = build_summary_context(state, recent_turns)
    if context is None:
        return

    response = client.chat.completions.create(
        model=model,
        messages=context,
        temperature=0,
    )
    state["summary"] = (response.choices[0].message.content or "").strip()


def run_turn(
    client: OpenAI,
    model: str,
    state: dict[str, Any],
    user_message: str,
    recent_turns: int | None = RECENT_TURNS,
    use_summary: bool = True,
    debug: bool = False,
) -> dict[str, Any]:
    """Process one complete user turn and return a structured outcome."""

    # current_turn contains only the actual conversation messages. Planner,
    # Executor, and tool messages remain in task state, not conversation history.
    current_turn: list[dict[str, Any]] = [
        {"role": "user", "content": user_message}
    ]
    state["current_task"] = create_task_state(goal=user_message)
    add_trace(state, "turn_started", goal=user_message)

    turn_status = "completed"

    # 계획·실행 중의 어떤 실패(LLM 예산 초과, JSON 검증 실패, replan 소진)도
    # 크래시가 아니라 '실패한 턴'으로 강등한다. 최종 답변은 항상 생성하되,
    # build_final_context가 실패 사실을 증거로 넘겨 지어내기를 막는다.
    try:
        create_plan(
            client,
            model,
            state,
            current_turn,
            recent_turns,
            use_summary,
            debug,
        )

        while True:
            step = get_current_plan_step(state)
            if step is None:
                break

            step_result = execute_plan_step(client, model, state, debug)
            step["result"] = step_result["content"]


            # If the step failed, no need to check progress
            # replan immediately and continue to the next iteration
            if step_result["status"] == "failed":
                failure_reason = step_result["content"]
                if state["current_task"]["replan_count"] >= MAX_REPLANS:
                    step["status"] = "failed"
                    turn_status = "failed"
                    add_trace(state, "replan_budget_exhausted", reason=failure_reason)
                    break
                replan(client, model, state, failure_reason, debug)
                continue

            progress = check_progress(client, model, state, step_result, debug)

            if progress["decision"] == "replan":
                if state["current_task"]["replan_count"] >= MAX_REPLANS:
                    step["status"] = "failed"
                    turn_status = "failed"
                    add_trace(state, "replan_budget_exhausted", reason=progress["reason"])
                    break
                replan(client, model, state, progress["reason"], debug)
                continue

            step["status"] = "completed"

            if progress["decision"] == "finish":
                state["current_task"]["current_plan_step_id"] = None
                break

            next_step = next_pending_plan_step(state)
            state["current_task"]["current_plan_step_id"] = (
                next_step["id"] if next_step else None
            )
            if next_step is None:
                break

    except (TurnBudgetExceeded, ValueError, RuntimeError) as error:
        turn_status = "failed"
        state["current_task"]["aborted_reason"] = str(error)
        add_trace(state, "turn_aborted", reason=str(error))

    # 최종 답변은 llm_call_count에 포함되지 않으므로
    # 예산이 소진된 뒤에도 항상 생성할 수 있다.
    final_answer = generate_final_answer(
        client,
        model,
        state,
        current_turn,
        recent_turns,
        use_summary,
        debug,
    )
    current_turn.append({"role": "assistant", "content": final_answer})
    add_trace(state, "turn_completed", status=turn_status)

    state["turns"].append(deepcopy(current_turn))
    if use_summary and recent_turns is not None:
        update_summary(client, model, state, recent_turns)

    return {
        "status": turn_status,
        "content": final_answer,
        "llm_calls": state["current_task"]["llm_call_count"],
        "replans": state["current_task"]["replan_count"],
        "trace": deepcopy(state["current_task"]["trace"]),
    }


def print_trace(trace: list[dict[str, Any]]) -> None:
    for index, event in enumerate(trace, start=1):
        print(f"{index:02d}. {json.dumps(event, ensure_ascii=False)}")


def create_client() -> OpenAI:
    import dotenv
    dotenv.load_dotenv()
    
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "https://agent.kau.ac.kr/v1"),
        api_key=os.getenv("KAU_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY")),
        timeout=120.0,
    )


if __name__ == "__main__":
    client = create_client()
    model = os.getenv("MODEL_NAME") or client.models.list().data[0].id
    state = create_state()

    question = (
        "2026년 해외 출장의 숙박비와 식비 한도, 그리고 출발 전 승인 요건을 "
        "근거 문서와 함께 정리해줘."
    )
    outcome = run_turn(client, model, state, question, debug=False)

    print(f"\n[{outcome['status']} / {outcome['llm_calls']} LLM calls / "
          f"{outcome['replans']} replans]")
    print("\n[Final Answer]\n")
    print(outcome["content"])


