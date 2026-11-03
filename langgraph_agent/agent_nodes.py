"""9개 Graph 노드와 분기 판정. 변경한 State 필드만 부분 update로 반환한다."""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Callable, Literal
from config import RECENT_TURNS, MAX_PLAN_STEPS, MAX_REPLANS, MAX_AGENT_LOOP_ITERATIONS
from agent_state import (AgentState, PlanStep, TaskState, create_plan_step, create_task_state,
                         get_plan_step, dependency_closure, artifacts_for_plan_steps,
                         compact_plan, next_ready_plan_step, turn_is_committed)
from context import (build_planner_context, build_executor_context, build_progress_context,
                     build_replanner_context, build_final_context, build_summary_context)
from llm import (TurnBudgetExceeded, call_json, call_completion, count_llm_call,
                 assistant_message_to_dict, show_context)


def start_turn_node(state: AgentState) -> dict[str, Any]:
    """대화 기록을 보존하면서 현재 사용자 입력의 실행 상태를 초기화한다.

    읽는 항목: current_turn의 사용자 메시지.
    갱신 항목: current_turn, task_state, step_result, progress,
               final_response, turn_status.
    보존 항목: turns, summary (반환 update에 포함하지 않는다).

    잘못된 입력은 작업을 시작하기 전에 ValueError로 알린다.
    LLM 호출과 Tool 실행은 수행하지 않는다.
    """
    input_messages = state.get("current_turn")
    if not isinstance(input_messages, list) or len(input_messages) != 1:
        raise ValueError(
            "current_turn must contain exactly one new user message."
        )

    message = input_messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        raise ValueError("The input message must have role='user'.")

    user_message = message.get("content")
    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("The user message content must be non-empty text.")

    # 새 객체이므로 trace를 추가해도 입력 State는 변경되지 않는다.
    task_state = create_task_state(goal=user_message)
    task_state["trace"].append({
        "event": "turn_started",
        "goal": user_message,
    })

    return {
        "current_turn": [
            {"role": "user", "content": user_message}
        ],
        "task_state": task_state,
        "step_result": None,
        "progress": None,
        "final_response": None,
        "turn_status": "running",
    }


def planner_node(
    state: AgentState,
    *,
    client: Any,
    model: str,
    recent_turns: int | None = RECENT_TURNS,
    use_summary: bool = True,
    debug: bool = False,
) -> dict[str, Any]:
    """최초 계획을 생성하고 변경된 task_state만 반환한다.

    client/model은 functools.partial로 미리 연결하여 사용한다.
    client는 OpenAI 호환 chat.completions.create 인터페이스를 제공한다.
    State에는 클라이언트나 모델 객체를 보관하지 않는다.

    기존 call_json의 재시도 동안 호출 횟수가 변하므로 task_state만
    deepcopy한다. 입력 State와 그 내부 객체는 직접 변경하지 않는다.

    성공: task_state를 반환하고 이후 review_plan_node로 진행한다.
    예상된 실패: 실패 원인과 turn_status='failed'를 반환한다.
    실패 후 finalizer로 이동하는 Edge는 Graph 구성 단계에서 연결한다.
    """
    if state["task_state"] is None:
        raise RuntimeError("Run start_turn_node before planner_node.")

    task_state = deepcopy(state["task_state"])

    try:
        context = build_planner_context(state, recent_turns, use_summary)
        show_context("PLANNER", context, debug)

        plan = call_json(
            client=client,
            model=model,
            messages=context,
            task_state=task_state,
            validate=lambda output: normalize_initial_plan(output.get("steps")),
        )

        task_state["plan"] = plan
        task_state["current_plan_step_id"] = plan[0]["id"]
        task_state["trace"].append({
            "event": "plan_created",
            "plan": compact_plan(plan),
        })

        return {"task_state": task_state}

    except (TurnBudgetExceeded, ValueError) as error:
        task_state["aborted_reason"] = str(error)
        task_state["trace"].append({
            "event": "turn_aborted",
            "phase": "planner",
            "reason": str(error),
        })
        return {
            "task_state": task_state,
            "turn_status": "failed",
        }


def review_plan_node(state: AgentState) -> dict[str, Any]:
    """계획을 보여주고 사용자 승인 또는 미실행 단계의 수정안을 받는다.

    응답: {"action": "approve"} 또는
          {"action": "edit", "edited_plan": [{"id": ..., "task": ...,
                                              "depends_on": [...]}, ...]}.

    edited_plan에는 수정 후 실행할 pending 단계 전체를 전달한다.
    기존 pending ID를 유지하거나, 기존 최대 ID보다 큰 ID로 단계를 추가한다.
    완료·대체·실패 단계는 수정할 수 없다. 삭제된 pending 단계는 replaced로 남긴다.

    interrupt를 한 번 호출한다. 응답 오류는 State에 저장하여 조건부 Edge로
    이 Node를 다시 호출한다. 같은 함수 안에서 interrupt 재입력 루프를 돌리지 않는다.
    """
    from langgraph.types import interrupt

    task_state = state["task_state"]
    if task_state is None or not any(
        step["status"] == "pending" for step in task_state["plan"]
    ):
        raise RuntimeError("Plan review requires at least one pending plan step.")

    # 중단 전에 하는 일은 읽기와 payload 생성뿐이다.
    review = interrupt({
        "type": "plan_review",
        "message": "계획을 승인하거나 미실행 단계의 수정안을 제출하세요.",
        "goal": task_state["goal"],
        "plan_revision": task_state["plan_revision"],
        "plan": compact_plan(task_state["plan"]),
        "editable_step_ids": [
            step["id"] for step in task_state["plan"]
            if step["status"] == "pending"
        ],
        "allowed_actions": ["approve", "edit"],
        "error": task_state["plan_review_error"],
    })

    try:
        reviewed_task = apply_plan_review(task_state, review)
    except ValueError as error:
        reviewed_task = deepcopy(task_state)
        reviewed_task["plan_review_error"] = str(error)
        reviewed_task["trace"].append({
            "event": "plan_review_invalid",
            "reason": str(error),
        })
        return {"task_state": reviewed_task}

    return {
        "task_state": reviewed_task,
        "step_result": None,
        "progress": None,
    }


def route_after_plan_review(
    state: AgentState,
) -> Literal["review_plan", "executor"]:
    """응답 오류가 있으면 재검토하고, 유효한 승인·수정 후에는 실행한다."""
    task_state = state["task_state"]
    if task_state is None:
        raise RuntimeError("Plan review routing requires task_state.")
    return (
        "review_plan"
        if task_state["plan_review_error"] is not None
        else "executor"
    )


def executor_node(
    state: AgentState,
    *,
    client: Any,
    model: str,
    tool_schemas: list[dict[str, Any]],
    tool_functions: dict[str, Callable[..., Any]],
    debug: bool = False,
) -> dict[str, Any]:
    """현재 계획 단계 하나를 실행하고 State update를 반환한다.

    원본 검색·메모리 함수와 그 schema는 partial로 외부에서 연결한다.
    Tool 함수는 원본과 동일하게 arguments만 받고 원시 검색 결과를 반환한다.
    반환 결과를 TaskState에 저장하는 일은 아래 execute_tool이 담당한다.

    step_result의 completed는 Executor loop가 결과를 반환했다는 의미다.
    PlanStep의 completed 판정은 다음 Progress Checker가 담당하므로
    정상 실행 직후의 단계 status는 in_progress로 유지한다.

    이 Node 전체가 checkpoint 단위다. 재실행하면 내부 Tool 호출도
    반복될 수 있으며, 외부 memory_write는 State 복사와 별개로 실행된다.
    """
    if state["task_state"] is None:
        raise RuntimeError("Executor requires task_state.")

    task_state = deepcopy(state["task_state"])
    step = get_plan_step(task_state, task_state["current_plan_step_id"])
    if step is None or step["status"] != "pending":
        raise RuntimeError("Executor requires a current pending plan step.")

    fatal_error = False
    try:
        step_result = execute_plan_step(
            client=client,
            model=model,
            task_state=task_state,
            step=step,
            tool_schemas=tool_schemas,
            tool_functions=tool_functions,
            debug=debug,
        )
    except (TurnBudgetExceeded, ValueError) as error:
        fatal_error = True
        task_state["aborted_reason"] = str(error)
        task_state["trace"].append({
            "event": "turn_aborted",
            "phase": "executor",
            "reason": str(error),
        })
        step_result = {"status": "failed", "content": str(error)}

    step["result"] = step_result["content"]
    if step_result["status"] == "failed":
        step["status"] = "failed"
    task_state["trace"].append({
        "event": "plan_step_executed",
        "plan_step_id": step["id"],
        "status": step_result["status"],
    })

    update: dict[str, Any] = {
        "task_state": task_state,
        "step_result": step_result,
        "progress": None,
    }
    if fatal_error:
        update["turn_status"] = "failed"
    return update


def progress_checker_node(
    state: AgentState,
    *,
    client: Any,
    model: str,
    debug: bool = False,
) -> dict[str, Any]:
    """정상 Executor 결과를 평가하여 단계 상태와 다음 실행 ID를 갱신한다.

    continue: 현재 단계를 완료로 확정하고 다음 실행 가능한 단계를 선택한다.
    finish: 현재 단계를 완료로 확정하고 현재 실행 ID를 비운다.
    replan: 현재 단계의 완료를 확정하지 않고 재계획 사유를 보존한다.

    Executor 실패는 route_after_executor가 Checker를 건너뛰게 한다.
    재계획 예산 소진이나 Checker 오류는 실패 상태를 반환하여 Finalizer로 보낸다.
    """
    if state["task_state"] is None:
        raise RuntimeError("Progress Checker requires task_state.")
    step_result = state["step_result"]
    if step_result is None or step_result.get("status") != "completed":
        raise RuntimeError("Progress Checker requires a completed Executor result.")

    task_state = deepcopy(state["task_state"])
    step = get_plan_step(task_state, task_state["current_plan_step_id"])
    if step is None or step["status"] != "in_progress":
        raise RuntimeError("Progress Checker requires a current in_progress step.")

    try:
        progress = check_progress(
            client, model, task_state, step, step_result, debug
        )

        if progress["decision"] == "replan":
            if task_state["replan_count"] >= MAX_REPLANS:
                step["status"] = "failed"
                task_state["aborted_reason"] = (
                    "The turn exhausted MAX_REPLANS: " + progress["reason"]
                )
                task_state["trace"].append({
                    "event": "replan_budget_exhausted",
                    "reason": progress["reason"],
                })
                return {
                    "task_state": task_state,
                    "progress": progress,
                    "turn_status": "failed",
                }
            # Replanner가 현재 단계와 pending 단계를 교체할 수 있도록 ID를 유지한다.
            return {"task_state": task_state, "progress": progress}

        step["status"] = "completed"
        if progress["decision"] == "finish":
            task_state["current_plan_step_id"] = None
        else:
            next_step = next_ready_plan_step(task_state["plan"])
            if next_step is None and any(
                item["status"] == "pending" for item in task_state["plan"]
            ):
                raise ValueError(
                    "Pending steps remain, but none has completed dependencies."
                )
            task_state["current_plan_step_id"] = (
                next_step["id"] if next_step is not None else None
            )

        return {"task_state": task_state, "progress": progress}

    except (TurnBudgetExceeded, ValueError) as error:
        step["status"] = "failed"
        task_state["aborted_reason"] = str(error)
        task_state["trace"].append({
            "event": "turn_aborted",
            "phase": "progress_checker",
            "reason": str(error),
        })
        return {
            "task_state": task_state,
            "progress": None,
            "turn_status": "failed",
        }


def route_after_executor(
    state: AgentState,
) -> Literal["progress_checker", "replanner", "finalizer"]:
    """치명적 실패는 최종 응답, 단계 실패는 재계획, 정상 결과는 Checker로 보낸다."""
    if state["turn_status"] == "failed":
        return "finalizer"
    step_result = state["step_result"]
    if step_result is None:
        raise RuntimeError("Executor routing requires step_result.")
    if step_result.get("status") == "failed":
        return "replanner"
    if step_result.get("status") == "completed":
        return "progress_checker"
    raise RuntimeError("Unknown Executor result status.")


def route_after_progress(
    state: AgentState,
) -> Literal["executor", "replanner", "finalizer"]:
    """State를 변경하지 않고 Checker가 확정한 상태로부터 다음 Node를 선택한다."""
    if state["turn_status"] == "failed":
        return "finalizer"
    progress = state["progress"]
    task_state = state["task_state"]
    if progress is None or task_state is None:
        raise RuntimeError("Progress routing requires progress and task_state.")
    if progress.get("decision") == "replan":
        return "replanner"
    if progress.get("decision") == "finish":
        return "finalizer"
    if progress.get("decision") == "continue":
        return (
            "executor"
            if task_state["current_plan_step_id"] is not None
            else "finalizer"
        )
    raise RuntimeError("Unknown progress decision.")


def replanner_node(
    state: AgentState,
    *,
    client: Any,
    model: str,
    debug: bool = False,
) -> dict[str, Any]:
    """완료 이력을 보존하고 미완료 계획을 교체한 뒤 사용자 검토로 보낸다.

    Checker의 replan 사유 또는 Executor의 실패 결과를 사용한다.
    새 계획을 검증한 후에만 기존 단계를 replaced로 바꾸고 횟수를 올린다.
    실패한 재계획은 이력을 교체하지 않고 Finalizer로 전달한다.
    """
    if state["task_state"] is None:
        raise RuntimeError("Replanner requires task_state.")
    reason = replanning_reason(state)
    task_state = deepcopy(state["task_state"])
    failed_step = get_plan_step(task_state, task_state["current_plan_step_id"])
    if failed_step is None or failed_step["status"] not in {"failed", "in_progress"}:
        raise RuntimeError("Replanner requires a failed or in_progress current step.")

    if task_state["replan_count"] >= MAX_REPLANS:
        failed_step["status"] = "failed"
        task_state["aborted_reason"] = "The turn exhausted MAX_REPLANS: " + reason
        task_state["trace"].append({
            "event": "replan_budget_exhausted",
            "reason": reason,
        })
        return {"task_state": task_state, "turn_status": "failed"}

    try:
        context = build_replanner_context(task_state, failed_step, reason)
        show_context("REPLANNER", context, debug)
        new_steps = call_json(
            client=client,
            model=model,
            messages=context,
            task_state=task_state,
            validate=lambda output: normalize_replanned_steps(
                output.get("remaining_steps"), task_state
            ),
        )

        failed_step["status"] = "replaced"
        for step in task_state["plan"]:
            if step["status"] == "pending":
                step["status"] = "replaced"
        task_state["plan"].extend(new_steps)
        task_state["current_plan_step_id"] = new_steps[0]["id"]
        task_state["plan_revision"] += 1
        task_state["replan_count"] += 1
        task_state["plan_review_error"] = None
        task_state["trace"].append({
            "event": "plan_revised",
            "reason": reason,
            "new_steps": [
                {"id": step["id"], "task": step["task"],
                 "depends_on": list(step["depends_on"])}
                for step in new_steps
            ],
        })
        return {"task_state": task_state, "step_result": None, "progress": None}

    except (TurnBudgetExceeded, ValueError) as error:
        failed_step["status"] = "failed"
        task_state["aborted_reason"] = str(error)
        task_state["trace"].append({
            "event": "turn_aborted",
            "phase": "replanner",
            "reason": str(error),
        })
        return {"task_state": task_state, "turn_status": "failed"}


def route_after_replanner(
    state: AgentState,
) -> Literal["review_plan", "finalizer"]:
    """재계획에 성공하면 사용자 검토, 실패하면 최종 응답으로 보낸다."""
    return "finalizer" if state["turn_status"] == "failed" else "review_plan"


def finalizer_node(
    state: AgentState,
    *,
    client: Any,
    model: str,
    recent_turns: int | None = RECENT_TURNS,
    use_summary: bool = True,
    debug: bool = False,
) -> dict[str, Any]:
    """완료 결과와 근거를 통합하여 final_response만 반환한다.

    실패한 turn에서도 실패 원인을 포함한 context로 응답을 생성한다.
    원본 정책에 따라 이 호출은 실행 예산 llm_call_count에 포함하지 않는다.
    State를 변경하지 않으므로 task_state를 복사할 필요가 없다.

    current_turn/turns 저장과 정상 turn의 completed 확정은 Commit이 담당한다.
    통신 오류나 빈 응답은 호출자에게 전파한다. Checkpoint에서 재개하면
    앞선 계획·실행을 반복하지 않고 이 Node를 다시 실행할 수 있다.
    """
    context = build_final_context(state, recent_turns, use_summary)
    show_context("FINAL ANSWER", context, debug)

    response = call_completion(
        client=client,
        model=model,
        messages=context,
        temperature=0,
    )
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Finalizer must return a non-empty text response.")

    return {"final_response": content.strip()}


def commit_turn_node(state: AgentState) -> dict[str, Any]:
    """실제 사용자 입력과 최종 응답 한 쌍을 현재 turn의 기록으로 확정한다.

    정상 turn은 completed로 바꾸고, 실패 turn은 failed를 유지한다.
    turn_completed trace를 대화 기록과 같은 update에 저장하므로,
    이미 확정된 State로 다시 호출해도 중복으로 추가하지 않는다.
    새 turn에서는 start_turn_node가 task_state와 trace를 새로 만든다.
    메시지 내용의 동일 여부로 서로 다른 turn을 중복 판정하지 않는다.
    """
    task_state = state["task_state"]
    if task_state is None:
        raise RuntimeError("Commit requires task_state.")
    if turn_is_committed(task_state):
        return {}
    if state["turn_status"] not in {"running", "failed"}:
        raise RuntimeError("Commit requires a running or failed turn.")

    final_response = state["final_response"]
    if not isinstance(final_response, str) or not final_response.strip():
        raise ValueError("Run finalizer_node before committing a turn.")
    messages = state["current_turn"]
    if (
        not isinstance(messages, list) or len(messages) != 1
        or not isinstance(messages[0], dict)
        or messages[0].get("role") != "user"
        or not isinstance(messages[0].get("content"), str)
        or not messages[0]["content"].strip()
    ):
        raise ValueError("Uncommitted current_turn must contain one user text message.")

    current_turn = deepcopy(messages)
    current_turn.append({"role": "assistant", "content": final_response})
    turn_status = "failed" if state["turn_status"] == "failed" else "completed"
    committed_task = deepcopy(task_state)
    committed_task["trace"].append({
        "event": "turn_completed",
        "status": turn_status,
    })

    return {
        "task_state": committed_task,
        "current_turn": current_turn,
        "turns": state["turns"] + [deepcopy(current_turn)],
        "turn_status": turn_status,
    }


def summary_node(
    state: AgentState,
    *,
    client: Any,
    model: str,
    recent_turns: int | None = RECENT_TURNS,
    use_summary: bool = True,
    debug: bool = False,
) -> dict[str, Any]:
    """기록이 확정된 turn 중 최근 N개를 제외한 대화를 다시 요약한다.

    원본처럼 turns 전체는 보관하고, 이전 turn들로 summary를 재생성한다.
    기존 summary를 입력에 합치지 않으므로 같은 내용을 누적 요약하지 않는다.
    recent_turns=None 또는 use_summary=False이면 LLM을 호출하지 않는다.
    recent_turns=0이면 보관된 모든 turn을 요약 대상으로 삼는다.

    이 호출도 실행 예산에 포함하지 않는다. 통신 오류는 전파하여
    Checkpoint에서 Summary만 재실행하게 한다. Commit은 이미 저장되어 있다.
    """
    if not use_summary or recent_turns is None:
        return {}
    task_state = state["task_state"]
    if task_state is None or not turn_is_committed(task_state):
        raise RuntimeError("Run commit_turn_node before summary_node.")

    context = build_summary_context(state, recent_turns)
    if context is None:
        return {}
    show_context("SUMMARY", context, debug)

    response = call_completion(
        client=client,
        model=model,
        messages=context,
        temperature=0,
    )
    content = response.choices[0].message.content
    # 보존할 정보가 없다면 빈 요약을 반환하는 것은 유효하다.
    if not isinstance(content, str):
        raise ValueError("Summary must return text; an empty string is allowed.")
    return {"summary": content.strip()}


def replanning_reason(state: AgentState) -> str:
    """Checker의 재계획 판정을 우선하고 Executor 실패를 별도로 처리한다."""
    progress = state["progress"]
    step_result = state["step_result"]
    if progress is not None and progress.get("decision") == "replan":
        reason = progress.get("reason")
    elif step_result is not None and step_result.get("status") == "failed":
        reason = step_result.get("content")
    else:
        raise RuntimeError("Replanner requires a replan decision or failed Executor result.")
    if not isinstance(reason, str) or not reason.strip():
        raise RuntimeError("Replanner requires a non-empty reason.")
    return reason.strip()


def normalize_replanned_steps(raw_steps: Any, task_state: TaskState) -> list[PlanStep]:
    """새 ID를 순차 부여하고 완료 단계·앞선 새 단계에 대한 의존성을 검증한다.

    모델의 ID는 참조용이며 실제 ID는 이력의 최대 ID 다음부터 부여한다.
    ID를 생략하면 해당 위치의 실제 ID를 참조용으로 사용한다.
    중복 ID나 완료 ID와의 충돌은 의존성 해석이 모호하므로 거부한다.
    """
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Replanner must return non-empty 'remaining_steps'.")
    if len(raw_steps) > MAX_PLAN_STEPS:
        raise ValueError(f"Replanner allows at most {MAX_PLAN_STEPS} new steps.")

    next_id = max((step["id"] for step in task_state["plan"]), default=0) + 1
    completed_ids = {
        step["id"] for step in task_state["plan"] if step["status"] == "completed"
    }
    model_id_to_assigned: dict[int, int] = {}
    for offset, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise ValueError("Each replanned step must be a JSON object.")
        model_id = raw_step.get("id", next_id + offset)
        if (
            type(model_id) is not int or model_id <= 0
            or model_id in model_id_to_assigned or model_id in completed_ids
        ):
            raise ValueError("New step IDs must be positive, unique, and not completed IDs.")
        model_id_to_assigned[model_id] = next_id + offset

    normalized: list[PlanStep] = []
    for offset, raw_step in enumerate(raw_steps):
        assigned_id = next_id + offset
        task = raw_step.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Replanned step {assigned_id} requires non-empty task text.")
        depends_on = raw_step.get("depends_on", [])
        if not isinstance(depends_on, list):
            raise ValueError(f"Replanned step {assigned_id} has invalid dependencies.")

        mapped_dependencies = []
        for dependency in depends_on:
            if type(dependency) is not int:
                raise ValueError("Replanned dependencies must be integer step IDs.")
            if dependency in completed_ids:
                mapped_dependencies.append(dependency)
            elif (
                dependency in model_id_to_assigned
                and model_id_to_assigned[dependency] < assigned_id
            ):
                mapped_dependencies.append(model_id_to_assigned[dependency])
            else:
                raise ValueError(
                    f"Replanned step {assigned_id} has invalid dependency {dependency!r}. "
                    "Use completed step IDs or IDs of earlier new steps."
                )
        normalized.append(
            create_plan_step(assigned_id, task.strip(), sorted(set(mapped_dependencies)))
        )
    return normalized


def check_progress(
    client: Any,
    model: str,
    task_state: TaskState,
    step: PlanStep,
    step_result: dict[str, str],
    debug: bool = False,
) -> dict[str, str]:
    """기존 Checker의 지시문과 JSON 재시도를 유지한다."""
    context = build_progress_context(task_state, step, step_result)
    show_context("PROGRESS CHECKER", context, debug)
    progress = call_json(
        client=client,
        model=model,
        messages=context,
        task_state=task_state,
        validate=validate_progress_output,
    )
    task_state["trace"].append({
        "event": "progress_checked",
        "plan_step_id": step["id"],
        **progress,
    })
    return progress


def validate_progress_output(output: dict[str, Any]) -> dict[str, str]:
    decision = output.get("decision")
    if not isinstance(decision, str) or decision not in {"continue", "replan", "finish"}:
        raise ValueError(f"Invalid progress decision: {decision}")
    reason = output.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError("Progress reason must be text.")
    reason = reason.strip()
    if decision == "replan" and not reason:
        raise ValueError("Replan decision must include a reason.")
    return {"decision": decision, "reason": reason}


def execute_plan_step(
    client: Any,
    model: str,
    task_state: TaskState,
    step: PlanStep,
    tool_schemas: list[dict[str, Any]],
    tool_functions: dict[str, Callable[..., Any]],
    debug: bool = False,
) -> dict[str, str]:
    """복사한 TaskState에서 기존 Tool-calling loop를 실행한다."""
    for dependency_id in dependency_closure(task_state, step):
        dependency = get_plan_step(task_state, dependency_id)
        if dependency is None or dependency["status"] != "completed":
            raise ValueError(f"Dependency {dependency_id} is not completed.")

    step["status"] = "in_progress"
    task_state["trace"].append({
        "event": "plan_step_started",
        "plan_step_id": step["id"],
        "task": step["task"],
        "depends_on": list(step["depends_on"]),
    })

    for iteration in range(1, MAX_AGENT_LOOP_ITERATIONS + 1):
        context = build_executor_context(task_state, step)
        show_context(
            f"EXECUTOR / PLAN STEP {step['id']} / ITERATION {iteration}",
            context,
            debug,
        )

        # Planner와 마찬가지로 Executor의 각 LLM 요청도 예산에 포함한다.
        count_llm_call(task_state)
        request: dict[str, Any] = {
            "model": model,
            "messages": context,
            "temperature": 0,
        }
        if tool_schemas:
            request.update(tools=tool_schemas, tool_choice="auto")
        response = call_completion(client=client, **request)
        assistant_message = response.choices[0].message
        step["messages"].append(assistant_message_to_dict(assistant_message))

        task_state["trace"].append({
            "event": "agent_loop_iteration",
            "plan_step_id": step["id"],
            "iteration": iteration,
            "tool_call_count": len(assistant_message.tool_calls or []),
        })

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
                result = execute_tool(
                    name, args, task_state, step, tool_functions
                )
                result_content = json.dumps(result, ensure_ascii=False)
                task_state["trace"].append({
                    "event": "tool_called",
                    "plan_step_id": step["id"],
                    "tool": name,
                    "arguments": args,
                    "result_summary": deepcopy(result),
                })
            except Exception as error:
                result = {"status": "error", "message": str(error)}
                result_content = json.dumps(result, ensure_ascii=False)
                task_state["trace"].append({
                    "event": "tool_error",
                    "plan_step_id": step["id"],
                    "tool": name,
                    "message": str(error),
                })

            step["messages"].append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_content,
            })

    return {
        "status": "failed",
        "content": (
            "The plan-step executor exceeded "
            f"{MAX_AGENT_LOOP_ITERATIONS} agent loop iterations."
        ),
    }


def execute_tool(
    name: str,
    args: dict[str, Any],
    task_state: TaskState,
    step: PlanStep,
    tool_functions: dict[str, Callable[..., Any]],
) -> Any:
    """등록된 원본 Tool을 실행하고 검색 근거를 복사본 State에 저장한다."""
    if name not in tool_functions:
        raise ValueError(f"Unknown or unconfigured tool: {name}")

    result = tool_functions[name](**args)
    # 부적합한 Tool 반환값은 State에 저장하기 전에 감지한다.
    json.dumps(result, ensure_ascii=False)

    if name == "search_documents":
        document_ids = save_retrieved_artifacts(task_state, step, result, "document")
        if not document_ids:
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
            "state_key": "task_state.retrieved_documents",
            "document_ids": document_ids,
        }

    if name == "search_memory":
        if not isinstance(result, dict) or "results" not in result:
            raise ValueError("search_memory must return an object with results.")
        memory_ids = save_retrieved_artifacts(
            task_state, step, result["results"], "memory"
        )
        return {
            "status": "saved",
            "state_key": "task_state.retrieved_memories",
            "memory_ids": memory_ids,
        }

    # memory_write 등 검색 이외의 Tool은 원본 반환값을 그대로 전달한다.
    return result


def save_retrieved_artifacts(
    task_state: TaskState,
    step: PlanStep,
    records: Any,
    kind: Literal["document", "memory"],
) -> list[str]:
    """검색 결과를 ID로 중복 제거하고 검색한 계획 단계의 출처를 기록한다."""
    if not isinstance(records, list):
        raise ValueError("Search results must be a list of objects with IDs.")

    # 전체 결과를 먼저 검증하여 잘못된 결과가 일부만 저장되는 것을 방지한다.
    checked: list[tuple[str, dict[str, Any]]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each search result must be an object.")
        artifact_id = record.get("id")
        if (
            type(artifact_id) not in (str, int)
            or not str(artifact_id).strip()
        ):
            raise ValueError("Each search result requires a string or integer ID.")
        checked.append((str(artifact_id), record))

    if kind == "document":
        store = task_state["retrieved_documents"]
        step_ids = step["document_ids"]
    else:
        store = task_state["retrieved_memories"]
        step_ids = step["memory_ids"]

    result_ids: list[str] = []
    for artifact_id, record in checked:
        if artifact_id not in store:
            store[artifact_id] = {
                **deepcopy(record),
                "retrieved_by_plan_steps": [],
            }
        provenance = store[artifact_id]["retrieved_by_plan_steps"]
        if step["id"] not in provenance:
            provenance.append(step["id"])
        if artifact_id not in step_ids:
            step_ids.append(artifact_id)
        if artifact_id not in result_ids:
            result_ids.append(artifact_id)
    return result_ids


def apply_plan_review(task_state: TaskState, review: Any) -> TaskState:
    """사용자 응답을 검증하고 새 TaskState를 만든다. 입력은 변경하지 않는다."""
    if not isinstance(review, dict):
        raise ValueError("Review must be an object with action='approve' or 'edit'.")

    action = review.get("action")
    if not isinstance(action, str) or action not in {"approve", "edit"}:
        raise ValueError("Review action must be 'approve' or 'edit'.")
    if action == "approve" and "edited_plan" in review:
        raise ValueError("Use action='edit' when submitting edited_plan.")

    reviewed_task = deepcopy(task_state)
    if action == "edit":
        edited_steps = normalize_edited_plan(review.get("edited_plan"), task_state)
        before = compact_plan(task_state["plan"])
        replacements = {step["id"]: step for step in edited_steps}
        rebuilt_plan: list[PlanStep] = []

        for step in reviewed_task["plan"]:
            if step["status"] != "pending":
                rebuilt_plan.append(step)
            elif step["id"] in replacements:
                rebuilt_plan.append(replacements.pop(step["id"]))
            else:
                step["status"] = "replaced"
                rebuilt_plan.append(step)

        # 기존 ID를 바꾸지 않으며, 새 단계는 역사상 최대 ID 뒤에 추가된다.
        rebuilt_plan.extend(replacements.values())
        reviewed_task["plan"] = rebuilt_plan
        reviewed_task["plan_revision"] += 1
        reviewed_task["trace"].append({
            "event": "plan_edited",
            "before": before,
            "after": compact_plan(rebuilt_plan),
            "plan_revision": reviewed_task["plan_revision"],
        })
    else:
        # 승인에서도 현재 pending 단계의 의존성이 유효한지 확인한다.
        normalize_edited_plan(
            [step for step in task_state["plan"] if step["status"] == "pending"],
            task_state,
        )

    next_step = next_ready_plan_step(reviewed_task["plan"])
    if next_step is None:
        raise ValueError("The reviewed plan has no executable pending step.")

    reviewed_task["current_plan_step_id"] = next_step["id"]
    reviewed_task["plan_review_error"] = None
    reviewed_task["trace"].append({
        "event": "plan_approved",
        "action": action,
        "plan_revision": reviewed_task["plan_revision"],
    })
    return reviewed_task


def normalize_edited_plan(raw_steps: Any, task_state: TaskState) -> list[PlanStep]:
    """완료 이력을 참조할 수 있는 미실행 단계의 전체 수정안을 검증한다."""
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("edited_plan must contain the complete non-empty pending plan.")
    if len(raw_steps) > MAX_PLAN_STEPS:
        raise ValueError(f"edited_plan allows at most {MAX_PLAN_STEPS} pending steps.")

    old_steps = {step["id"]: step for step in task_state["plan"]}
    maximum_id = max(old_steps, default=0)
    available_dependencies = {
        step["id"] for step in task_state["plan"] if step["status"] == "completed"
    }
    normalized: list[PlanStep] = []
    previous_id = 0

    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            raise ValueError("Each edited step must be an object.")

        step_id = raw_step.get("id")
        if type(step_id) is not int or step_id <= previous_id:
            raise ValueError("Edited step IDs must be positive, unique, and ascending.")
        if step_id in old_steps:
            if old_steps[step_id]["status"] != "pending":
                raise ValueError(f"Only pending steps can be edited: {step_id}.")
        elif step_id <= maximum_id:
            raise ValueError(f"A new step ID must be greater than {maximum_id}.")

        task = raw_step.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Edited step {step_id} requires non-empty task text.")

        depends_on = raw_step.get("depends_on", [])
        if not isinstance(depends_on, list) or not all(
            type(dependency) is int
            and dependency < step_id
            and dependency in available_dependencies
            for dependency in depends_on
        ):
            raise ValueError(
                f"Step {step_id} dependencies must be completed IDs or earlier edited IDs."
            )

        normalized.append(
            create_plan_step(step_id, task.strip(), sorted(set(depends_on)))
        )
        available_dependencies.add(step_id)
        previous_id = step_id

    return normalized


def normalize_initial_plan(raw_steps: Any) -> list[PlanStep]:
    """계획을 검증하고 순차 ID와 기존 초기값을 부여한다."""
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Planner output must contain a non-empty 'steps' list.")
    if len(raw_steps) > MAX_PLAN_STEPS:
        raise ValueError(
            f"Planner returned {len(raw_steps)} steps but at most "
            f"{MAX_PLAN_STEPS} are allowed. Merge related subtasks into "
            "fewer, self-contained steps."
        )

    normalized: list[PlanStep] = []
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            raise ValueError("Each plan step must be a JSON object.")

        task = raw_step.get("task", "")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Plan step {index} has no valid task text.")

        depends_on = raw_step.get("depends_on", [])
        if not isinstance(depends_on, list) or not all(
            type(item) is int and 1 <= item < index for item in depends_on
        ):
            raise ValueError(f"Plan step {index} has invalid dependencies.")

        normalized.append(
            create_plan_step(index, task.strip(), sorted(set(depends_on)))
        )
    return normalized


def route_after_planner(
    state: AgentState,
) -> Literal["review_plan", "finalizer"]:
    """최초 계획에 성공하면 검토, 실패하면 최종 응답으로 보낸다."""
    if state["turn_status"] == "failed":
        return "finalizer"
    task_state = state["task_state"]
    if task_state is None or not task_state["plan"]:
        raise RuntimeError("Planner routing requires a generated plan.")
    return "review_plan"

