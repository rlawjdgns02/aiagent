"""Graph nodes and routers for the Plan–Execute–Replan Agent.

노드
    planner           계획 생성
    plan_review       사용자 검토 (interrupt)
    executor          plan step 실행
    progress_checker  진행 판단
    replanner         재계획
    final_answer      최종 응답
    update_summary    요약 갱신

라우터
    State를 읽고 다음 노드 이름만 반환한다. State를 바꾸지 않는다.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import interrupt
from openai import OpenAI

from contexts import (
    build_executor_context,
    build_final_context,
    build_planner_context,
    build_progress_context,
    build_replanner_context,
    build_summary_context,
)
from llm import (
    LLMCallBudget,
    TurnBudgetExceeded,
    assistant_message_to_dict,
    call_json,
    show_context,
)
from state import (
    AgentState,
    PlanStep,
    ProgressResult,
    StepResult,
    TaskState,
    compact_plan,
    get_plan_step,
    next_pending_plan_step,
    normalize_initial_plan,
    normalize_replanned_steps,
    trace_event,
    update_plan_step,
)
from tools import TOOLS, execute_tool


RECENT_TURNS = 2
MAX_REPLANS = 2
MAX_AGENT_LOOP_ITERATIONS = 6

# 노드에서 이 예외가 나면 턴을 실패로 기록하고 final_answer로 보낸다.
AGENT_ERRORS = (TurnBudgetExceeded, ValueError, RuntimeError)


@dataclass
class AgentContext:
    """Runtime context: invoke할 때마다 넘기는 실행 도구와 옵션.

    State가 아니므로 checkpoint에 저장되지 않는다.
    그래서 interrupt 후 재개할 때도 같은 context를 다시 넘겨야 한다.
    """

    client: OpenAI
    model: str
    recent_turns: int | None = RECENT_TURNS
    use_summary: bool = True
    debug: bool = False


def aborted_task(task: TaskState, error: Exception, llm_call_count: int) -> TaskState:
    """예외를 '실패한 턴'으로 바꾼 TaskState. 라우터가 aborted_reason을 보고 final_answer로 보낸다."""

    return {
        **task,
        "turn_status": "failed",
        "aborted_reason": str(error),
        "llm_call_count": llm_call_count,
        "trace": [*task["trace"], trace_event("turn_aborted", reason=str(error))],
    }


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def planner(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """대화 맥락과 현재 질문 → 실행 가능한 plan step 목록.

    변경: current_task의 plan, current_plan_step_id, llm_call_count, trace
    """

    ctx = runtime.context
    task = state["current_task"]
    budget = LLMCallBudget(task["llm_call_count"])

    context = build_planner_context(
        state,
        state["current_turn"],
        ctx.recent_turns,
        ctx.use_summary,
    )
    show_context("PLANNER", context, ctx.debug)

    try:
        plan = call_json(
            ctx.client, ctx.model, context, budget,
            validate=lambda output: normalize_initial_plan(output.get("steps")),
        )
    except AGENT_ERRORS as error:
        return {"current_task": aborted_task(task, error, budget.used)}

    return {
        "current_task": {
            **task,
            "plan": plan,
            "current_plan_step_id": plan[0]["id"],
            "llm_call_count": budget.used,
            "trace": [*task["trace"], trace_event("plan_created", plan=compact_plan(plan))],
        }
    }


# ---------------------------------------------------------------------------
# Plan review: interrupt로 사용자 승인·수정 받기
# ---------------------------------------------------------------------------

def plan_review(state: AgentState) -> dict[str, Any]:
    """계획을 사용자에게 보여주고 멈춘 뒤, 승인 또는 수정된 계획을 받는다.

    재개 방법: Command(resume={"action": "approve"})
              Command(resume={"action": "edit", "steps": [{"instruction": ..., "depends_on": [...]}, ...]})

    재개하면 이 노드는 처음부터 다시 실행되고, 두 번째 실행에서 interrupt()는
    멈추지 않고 resume 값을 반환한다. 그래서
    - 계획 생성 LLM 호출은 planner 노드에 두고, 이 노드에서는 하지 않는다.
    - interrupt() 앞에는 다시 실행되어도 결과가 같은 계산만 둔다.

    변경: 승인 → trace
          수정 → plan, current_plan_step_id, trace
          잘못된 수정 → review_error (라우터가 이 노드로 다시 보내 재검토를 요청)
    """

    task = state["current_task"]

    review = interrupt(
        {
            "goal": task["goal"],
            "plan": compact_plan(task["plan"]),
            "error": task["review_error"],
        }
    )

    # ---- 여기부터는 Command(resume=...)로 재개된 실행에서만 실행된다 ----
    action = review.get("action") if isinstance(review, dict) else None

    if action == "approve":
        return {
            "current_task": {
                **task,
                "review_error": None,
                "trace": [*task["trace"], trace_event("plan_reviewed", action="approve")],
            }
        }

    if action == "edit":
        try:
            # 사람이 수정한 계획도 Planner 출력과 같은 규칙으로 검증한다.
            plan = normalize_initial_plan(review.get("steps"))
        except ValueError as error:
            return {"current_task": {**task, "review_error": str(error)}}

        return {
            "current_task": {
                **task,
                "plan": plan,
                "current_plan_step_id": plan[0]["id"],
                "review_error": None,
                "trace": [
                    *task["trace"],
                    trace_event("plan_reviewed", action="edit", plan=compact_plan(plan)),
                ],
            }
        }

    return {
        "current_task": {
            **task,
            "review_error": "The review must be {'action': 'approve'} or {'action': 'edit', 'steps': [...]}.",
        }
    }


# ---------------------------------------------------------------------------
# Executor: 현재 plan step 하나를 도구 호출 agent loop로 수행
# ---------------------------------------------------------------------------

def execute_plan_step(
    ctx: AgentContext,
    work_state: AgentState,
    step: PlanStep,
) -> StepResult:
    """현재 plan step 하나를 LLM ↔ Tool 호출 반복으로 실행한다.

    work_state와 step은 executor 노드가 만든 작업용 사본이다.
    execute_tool이 넘겨받은 state를 직접 고치기 때문에 사본을 넘긴다.
    """

    trace = work_state["current_task"]["trace"]

    for iteration in range(1, MAX_AGENT_LOOP_ITERATIONS + 1):
        # 매 iteration마다 다시 만든다: 방금 검색한 문서가 Step Context에 들어가야 하므로.
        context = build_executor_context(work_state, step)
        show_context(
            f"EXECUTOR / PLAN STEP {step['id']} / ITERATION {iteration}",
            context,
            ctx.debug,
        )

        # Executor 호출은 llm_call_count에 포함하지 않는다.
        response = ctx.client.chat.completions.create(
            model=ctx.model,
            messages=context,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        assistant_message = response.choices[0].message
        step["messages"].append(assistant_message_to_dict(assistant_message))
        trace.append(
            trace_event(
                "agent_loop_iteration",
                plan_step_id=step["id"],
                iteration=iteration,
                tool_call_count=len(assistant_message.tool_calls or []),
            )
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
                result = execute_tool(name, args, work_state, step)
                trace.append(
                    trace_event(
                        "tool_called",
                        plan_step_id=step["id"],
                        tool=name,
                        arguments=args,
                        result_summary=result,
                    )
                )
            except Exception as error:
                result = {"status": "error", "message": str(error)}
                trace.append(
                    trace_event(
                        "tool_error",
                        plan_step_id=step["id"],
                        tool=name,
                        message=str(error),
                    )
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


def executor(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """현재 plan step → 그 step의 실행 결과(completed/failed).

    변경: current_task의 plan(현재 step의 status·messages·result·ids),
          retrieved_documents, retrieved_memories, trace, step_result, progress(초기화)
    """

    ctx = runtime.context
    task = state["current_task"]

    # 입력 State를 지키기 위해, 이 노드가 고칠 항목만 복사한 작업용 task를 만든다.
    work_task: TaskState = {
        **task,
        "plan": deepcopy(task["plan"]),
        "retrieved_documents": deepcopy(task["retrieved_documents"]),
        "retrieved_memories": deepcopy(task["retrieved_memories"]),
        "trace": list(task["trace"]),
    }
    work_state: AgentState = {**state, "current_task": work_task}
    step = get_plan_step(work_task["plan"], work_task["current_plan_step_id"])

    step["status"] = "in_progress"
    work_task["trace"].append(
        trace_event(
            "plan_step_started",
            plan_step_id=step["id"],
            instruction=step["instruction"],
            depends_on=step["depends_on"],
        )
    )

    try:
        step_result = execute_plan_step(ctx, work_state, step)
    except AGENT_ERRORS as error:
        return {
            "current_task": aborted_task(work_task, error, work_task["llm_call_count"])
        }

    step["result"] = step_result["content"]

    return {
        "current_task": {
            **work_task,
            "step_result": step_result,
            "progress": None,  # 이전 step의 진행 판단을 지운다
        }
    }


# ---------------------------------------------------------------------------
# Progress Checker
# ---------------------------------------------------------------------------

def validate_progress_output(output: dict[str, Any]) -> ProgressResult:
    decision = output.get("decision")
    if decision not in {"continue", "replan", "finish"}:
        raise ValueError(f"Invalid progress decision: {decision}")
    reason = str(output.get("reason", "")).strip()
    if decision == "replan" and not reason:
        raise ValueError("Replan decision must include a reason.")
    return {"decision": decision, "reason": reason}


def progress_checker(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> dict[str, Any]:
    """현재 step의 실행 결과 → continue / replan / finish 결정.

    continue·finish이면 step 완료 처리와 다음 step 선택까지 한다.

    변경: progress, llm_call_count, trace
          + continue·finish → plan(현재 step completed), current_plan_step_id
    """

    ctx = runtime.context
    task = state["current_task"]
    budget = LLMCallBudget(task["llm_call_count"])
    step = get_plan_step(task["plan"], task["current_plan_step_id"])

    context = build_progress_context(state, step, task["step_result"])
    show_context("PROGRESS CHECKER", context, ctx.debug)

    try:
        progress = call_json(
            ctx.client, ctx.model, context, budget,
            validate=validate_progress_output,
        )
    except AGENT_ERRORS as error:
        return {"current_task": aborted_task(task, error, budget.used)}

    trace = [
        *task["trace"],
        trace_event("progress_checked", plan_step_id=step["id"], **progress),
    ]

    if progress["decision"] == "replan":
        # step 상태는 replanner가 바꾼다.
        return {
            "current_task": {
                **task,
                "progress": progress,
                "llm_call_count": budget.used,
                "trace": trace,
            }
        }

    plan = update_plan_step(task["plan"], step["id"], status="completed")
    if progress["decision"] == "finish":
        next_plan_step_id = None
    else:
        next_step = next_pending_plan_step(plan)
        next_plan_step_id = next_step["id"] if next_step else None

    return {
        "current_task": {
            **task,
            "plan": plan,
            "current_plan_step_id": next_plan_step_id,
            "progress": progress,
            "llm_call_count": budget.used,
            "trace": trace,
        }
    }


# ---------------------------------------------------------------------------
# Replanner
# ---------------------------------------------------------------------------

def replanner(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """실패했거나 부족했던 step → 그 step과 남은 step을 대체할 새 step 목록.

    두 곳에서 온다: executor(step 실패), progress_checker(replan 결정).

    변경: 한도 초과 → plan(현재 step failed), turn_status, trace
          재계획     → plan, current_plan_step_id, plan_revision, replan_count, llm_call_count, trace
    """

    ctx = runtime.context
    task = state["current_task"]
    failed_step = get_plan_step(task["plan"], task["current_plan_step_id"])

    if task["step_result"]["status"] == "failed":
        reason = task["step_result"]["content"]
    else:
        reason = task["progress"]["reason"]

    if task["replan_count"] >= MAX_REPLANS:
        return {
            "current_task": {
                **task,
                "plan": update_plan_step(task["plan"], failed_step["id"], status="failed"),
                "turn_status": "failed",
                "trace": [*task["trace"], trace_event("replan_budget_exhausted", reason=reason)],
            }
        }

    budget = LLMCallBudget(task["llm_call_count"])
    context = build_replanner_context(state, failed_step, reason)
    show_context("REPLANNER", context, ctx.debug)

    try:
        new_steps = call_json(
            ctx.client, ctx.model, context, budget,
            validate=lambda output: normalize_replanned_steps(
                output.get("remaining_steps"), task["plan"]
            ),
        )
    except AGENT_ERRORS as error:
        return {"current_task": aborted_task(task, error, budget.used)}

    # 실패한 step과 아직 실행하지 않은 step을 replaced로 바꾸고, 새 step을 뒤에 붙인다.
    plan = [
        {**step, "status": "replaced"}
        if step["id"] == failed_step["id"] or step["status"] == "pending"
        else step
        for step in task["plan"]
    ] + new_steps

    return {
        "current_task": {
            **task,
            "plan": plan,
            "current_plan_step_id": new_steps[0]["id"],
            "plan_revision": task["plan_revision"] + 1,
            "replan_count": task["replan_count"] + 1,
            "llm_call_count": budget.used,
            "trace": [
                *task["trace"],
                trace_event(
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
                ),
            ],
        }
    }


# ---------------------------------------------------------------------------
# Final answer and summary
# ---------------------------------------------------------------------------

def final_answer(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """작업 결과와 증거 → 사용자에게 보여줄 최종 답변. 이번 턴을 turns에 저장한다.

    변경: current_turn(답변 추가), turns(이번 턴 누적), current_task의 trace
    """

    ctx = runtime.context
    task = state["current_task"]

    context = build_final_context(
        state,
        state["current_turn"],
        ctx.recent_turns,
        ctx.use_summary,
    )
    show_context("FINAL ANSWER", context, ctx.debug)

    # 최종 답변은 llm_call_count에 포함되지 않으므로
    # 예산이 소진된 뒤에도 항상 생성할 수 있다.
    response = ctx.client.chat.completions.create(
        model=ctx.model,
        messages=context,
        temperature=0,
    )
    answer = (response.choices[0].message.content or "").strip()
    current_turn = [*state["current_turn"], {"role": "assistant", "content": answer}]

    return {
        "current_turn": current_turn,
        "turns": [current_turn],  # operator.add reducer: 기존 turns 뒤에 이번 턴을 붙인다
        "current_task": {
            **task,
            "trace": [*task["trace"], trace_event("turn_completed", status=task["turn_status"])],
        },
    }


def update_summary(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> dict[str, Any]:
    """최근 recent_turns개보다 오래된 턴 → 요약. 요약할 턴이 없으면 아무것도 바꾸지 않는다.

    변경: summary
    """

    ctx = runtime.context
    if not ctx.use_summary or ctx.recent_turns is None:
        return {}

    context = build_summary_context(state, ctx.recent_turns)
    if context is None:
        return {}

    response = ctx.client.chat.completions.create(
        model=ctx.model,
        messages=context,
        temperature=0,
    )
    return {"summary": (response.choices[0].message.content or "").strip()}


# ---------------------------------------------------------------------------
# Routers (conditional edges)
# ---------------------------------------------------------------------------

def route_after_planner(state: AgentState) -> str:
    if state["current_task"]["aborted_reason"]:
        return "final_answer"
    return "plan_review"


def route_after_review(state: AgentState) -> str:
    if state["current_task"]["review_error"]:
        return "plan_review"        # 잘못된 수정 → 다시 검토 요청
    return "executor"


def route_after_executor(state: AgentState) -> str:
    task = state["current_task"]
    if task["aborted_reason"]:
        return "final_answer"
    if task["step_result"]["status"] == "failed":
        return "replanner"          # 실패한 step은 진행 판단 없이 바로 재계획
    return "progress_checker"


def route_after_progress(state: AgentState) -> str:
    task = state["current_task"]
    if task["aborted_reason"]:
        return "final_answer"
    if task["progress"]["decision"] == "replan":
        return "replanner"
    if task["current_plan_step_id"] is None:
        return "final_answer"       # finish, 또는 남은 step 없음
    return "executor"               # continue: 다음 step 실행


def route_after_replanner(state: AgentState) -> str:
    task = state["current_task"]
    if task["aborted_reason"] or task["turn_status"] == "failed":
        return "final_answer"       # 중단, 또는 재계획 한도 초과
    return "executor"
