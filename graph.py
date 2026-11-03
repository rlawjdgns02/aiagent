"""Plan–Execute–Replan Agent orchestration with LangGraph.

Module boundary
---------------
- tools.py: tool definitions and execution
- contexts.py: phase-specific context projection
- state.py: State schema, constructors, plan validation
- llm.py: LLM call helpers
- nodes.py: nodes (phases) and routers (branches)
- graph.py: graph wiring, checkpointer, turn start/resume

Graph
-----
START → planner → plan_review ⇄ (잘못된 수정이면 다시 plan_review)
                      ↓
                  executor → progress_checker → executor (continue)
                      ↓              ↓
                      └──→ replanner ←┘ (step 실패 / replan)
                              ↓
                          executor
planner·executor·progress_checker·replanner는 중단·종료 시 final_answer로 간다.
final_answer → update_summary → END
"""

from __future__ import annotations

import json
import os
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from openai import OpenAI

from nodes import (
    AgentContext,
    executor,
    final_answer,
    plan_review,
    planner,
    progress_checker,
    replanner,
    route_after_executor,
    route_after_planner,
    route_after_progress,
    route_after_replanner,
    route_after_review,
    update_summary,
)
from state import AgentState, create_state, create_task_state, trace_event


# 노드 실행 1회가 LangGraph super-step 1회다. 기본 한도(25)는
# step이 많거나 재계획이 일어난 턴에서 모자랄 수 있어 늘린다.
RECURSION_LIMIT = 100


def build_graph(checkpointer: Any | None = None) -> CompiledStateGraph:
    builder = StateGraph(AgentState, context_schema=AgentContext)

    builder.add_node("planner", planner)
    builder.add_node("plan_review", plan_review)
    builder.add_node("executor", executor)
    builder.add_node("progress_checker", progress_checker)
    builder.add_node("replanner", replanner)
    builder.add_node("final_answer", final_answer)
    builder.add_node("update_summary", update_summary)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner", route_after_planner, ["plan_review", "final_answer"]
    )
    builder.add_conditional_edges(
        "plan_review", route_after_review, ["executor", "plan_review"]
    )
    builder.add_conditional_edges(
        "executor",
        route_after_executor,
        ["progress_checker", "replanner", "final_answer"],
    )
    builder.add_conditional_edges(
        "progress_checker",
        route_after_progress,
        ["executor", "replanner", "final_answer"],
    )
    builder.add_conditional_edges(
        "replanner", route_after_replanner, ["executor", "final_answer"]
    )
    builder.add_edge("final_answer", "update_summary")
    builder.add_edge("update_summary", END)

    # checkpointer는 노드가 끝날 때마다 State를 thread_id별로 저장한다.
    # 같은 thread_id로 다시 invoke하면 이전 대화(turns, summary)가 이어지고,
    # interrupt로 멈춘 실행도 그 지점부터 재개할 수 있다.
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


# ---------------------------------------------------------------------------
# Turn lifecycle
# ---------------------------------------------------------------------------

def make_config(thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }


def start_turn(
    app: CompiledStateGraph,
    context: AgentContext,
    thread_id: str,
    user_message: str,
) -> dict[str, Any]:
    """새 사용자 메시지로 턴을 시작한다. 계획 검토에서 멈추면 검토 요청을 반환한다."""

    config = make_config(thread_id)
    snapshot = app.get_state(config)

    # 계획 검토 대기 중인 thread에 새 입력을 넣으면 LangGraph는 멈춘 실행을 버리고
    # START부터 다시 시작한다. 검토 중이던 질문이 기록 없이 사라지므로 막는다.
    if snapshot.interrupts:
        raise ValueError(
            f"Thread {thread_id!r} is waiting for a plan review. Call resume_turn first."
        )

    # 이번 턴의 current_turn과 current_task를 새로 만든다.
    # turns와 summary는 입력에 넣지 않으므로 checkpoint에 저장된 값이 유지된다.
    task = create_task_state(goal=user_message)
    task = {**task, "trace": [trace_event("turn_started", goal=user_message)]}
    graph_input: dict[str, Any] = {
        "current_turn": [{"role": "user", "content": user_message}],
        "current_task": task,
    }

    # 새 대화(thread)의 첫 턴이면 대화 State도 초기화한다.
    if not snapshot.values:
        graph_input = {**create_state(), **graph_input}

    app.invoke(graph_input, config, context=context)
    return get_outcome(app, thread_id)


def resume_turn(
    app: CompiledStateGraph,
    context: AgentContext,
    thread_id: str,
    review: dict[str, Any],
) -> dict[str, Any]:
    """멈춰 있는 계획 검토에 사용자 결정을 넘기고 실행을 이어간다.

    review: {"action": "approve"}
            {"action": "edit", "steps": [{"instruction": ..., "depends_on": [...]}, ...]}
    """

    config = make_config(thread_id)
    if not app.get_state(config).interrupts:
        raise ValueError(f"Thread {thread_id!r} is not waiting for a plan review.")

    app.invoke(Command(resume=review), config, context=context)
    return get_outcome(app, thread_id)


def get_outcome(app: CompiledStateGraph, thread_id: str) -> dict[str, Any]:
    """저장된 State로 결과를 만든다: 검토 대기 중이면 검토 요청, 끝났으면 최종 답변과 실행 정보."""

    snapshot = app.get_state(make_config(thread_id))

    if snapshot.interrupts:
        return {
            "status": "awaiting_review",
            "review_request": snapshot.interrupts[0].value,
        }

    values = snapshot.values
    task = values["current_task"]
    return {
        "status": task["turn_status"],
        "content": values["current_turn"][-1]["content"],
        "llm_calls": task["llm_call_count"],
        "replans": task["replan_count"],
        "trace": task["trace"],
    }


# ---------------------------------------------------------------------------
# Printing and client
# ---------------------------------------------------------------------------

def print_trace(trace: list[dict[str, Any]]) -> None:
    for index, event in enumerate(trace, start=1):
        print(f"{index:02d}. {json.dumps(event, ensure_ascii=False)}")


def print_outcome(outcome: dict[str, Any]) -> None:
    if outcome["status"] == "awaiting_review":
        request = outcome["review_request"]
        print(f"[계획 검토 요청] {request['goal']}\n")
        if request["error"]:
            print(f"! 이전 수정이 거부됨: {request['error']}\n")
        for step in request["plan"]:
            print(f"{step['id']}. {step['instruction']}  (depends_on={step['depends_on']})")
        return

    print(f"\n[{outcome['status']} / {outcome['llm_calls']} LLM calls / "
          f"{outcome['replans']} replans]")
    print("\n[Final Answer]\n")
    print(outcome["content"])
    print("\n[Trace]\n")
    print_trace(outcome["trace"])


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
    context = AgentContext(client=client, model=model)
    app = build_graph()

    question = (
        "2026년 해외 출장의 숙박비와 식비 한도, 그리고 출발 전 승인 요건을 "
        "근거 문서와 함께 정리해줘."
    )
    outcome = start_turn(app, context, "demo", question)
    print_outcome(outcome)

    # 데모에서는 계획을 그대로 승인한다.
    outcome = resume_turn(app, context, "demo", {"action": "approve"})
    print_outcome(outcome)
