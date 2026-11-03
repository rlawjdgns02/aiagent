"""구현한 9개 Node를 연결하여 실행 가능한 LangGraph를 구성한다.

agent_state.py, agent_nodes.py와 같은 디렉터리에 둔다.
client와 검색·메모리 Tool은 호출자가 전달하며 State에는 넣지 않는다.
Graph 생성만으로 LLM 호출이나 Tool 실행이 발생하지 않는다.

호출 계약:
- Graph는 한 번 구성한 뒤 재사용한다.
- 첫 질문: create_state()의 초기값과 current_turn을 함께 전달한다.
- 다음 질문: 같은 thread_id에서 새 current_turn만 전달한다.
- 검토 응답: 같은 thread_id에서 Command(resume=...)로 재개한다.
- Node 오류 후 재시도: 같은 thread_id에서 graph.invoke(None, config).

기본 InMemorySaver는 현재 프로세스에서만 checkpoint를 보관한다.
다른 저장 방식을 쓰려면 checkpointer 인자로 전달한다.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from config import RECENT_TURNS

from agent_nodes import (
    route_after_planner,
    commit_turn_node,
    executor_node,
    finalizer_node,
    planner_node,
    progress_checker_node,
    replanner_node,
    review_plan_node,
    route_after_executor,
    route_after_plan_review,
    route_after_progress,
    route_after_replanner,
    start_turn_node,
    summary_node,
)
from agent_state import AgentState



def build_agent_graph(
    *,
    client: Any,
    model: str,
    tool_schemas: list[dict[str, Any]],
    tool_functions: dict[str, Callable[..., Any]],
    checkpointer: BaseCheckpointSaver | None = None,
    recent_turns: int | None = RECENT_TURNS,
    use_summary: bool = True,
    debug: bool = False,
) -> CompiledStateGraph:
    """외부 객체를 Node에 연결하고 checkpoint가 있는 Graph를 반환한다."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string.")
    if recent_turns is not None and (
        type(recent_turns) is not int or recent_turns < 0
    ):
        raise ValueError("recent_turns must be None or a non-negative integer.")
    if checkpointer is None:
        checkpointer = InMemorySaver()

    builder = StateGraph(AgentState)
    llm_options = {"client": client, "model": model, "debug": debug}
    conversation_options = {
        "recent_turns": recent_turns,
        "use_summary": use_summary,
    }

    # (1)~(9) Node 등록. 외부 객체는 partial에 연결하고 State에는 저장하지 않는다.
    builder.add_node("start_turn", start_turn_node)
    builder.add_node(
        "planner",
        partial(planner_node, **llm_options, **conversation_options),
    )
    builder.add_node("review_plan", review_plan_node)
    builder.add_node(
        "executor",
        partial(
            executor_node,
            **llm_options,
            tool_schemas=tool_schemas,
            tool_functions=tool_functions,
        ),
    )
    builder.add_node(
        "progress_checker", partial(progress_checker_node, **llm_options)
    )
    builder.add_node("replanner", partial(replanner_node, **llm_options))
    builder.add_node(
        "finalizer",
        partial(finalizer_node, **llm_options, **conversation_options),
    )
    builder.add_node("commit_turn", commit_turn_node)
    builder.add_node(
        "summary",
        partial(summary_node, **llm_options, **conversation_options),
    )

    # 고정 Edge: 새 turn 시작과 최종 응답 이후의 저장·요약.
    builder.add_edge(START, "start_turn")
    builder.add_edge("start_turn", "planner")
    builder.add_edge("finalizer", "commit_turn")
    builder.add_edge("commit_turn", "summary")
    builder.add_edge("summary", END)

    # 조건부 Edge: Router가 반환한 이름을 실제 Node에 연결한다.
    # 승인 또는 수정에 성공한 경우에만 Executor로 이동한다.
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {"review_plan": "review_plan", "finalizer": "finalizer"},
    )
    builder.add_conditional_edges(
        "review_plan",
        route_after_plan_review,
        {"review_plan": "review_plan", "executor": "executor"},
    )
    builder.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "progress_checker": "progress_checker",
            "replanner": "replanner",
            "finalizer": "finalizer",
        },
    )
    builder.add_conditional_edges(
        "progress_checker",
        route_after_progress,
        {
            "executor": "executor",
            "replanner": "replanner",
            "finalizer": "finalizer",
        },
    )
    builder.add_conditional_edges(
        "replanner",
        route_after_replanner,
        {"review_plan": "review_plan", "finalizer": "finalizer"},
    )

    # review_plan_node 내부 interrupt와 checkpointer로 중단·재개한다.
    return builder.compile(checkpointer=checkpointer)

