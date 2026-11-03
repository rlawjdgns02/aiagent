"""대화·작업 State의 타입, 초기값, 상태를 조회하는 순수 함수."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


PlanStepStatus = Literal[
    "pending", "in_progress", "completed", "failed", "replaced"
]
TurnStatus = Literal["idle", "running", "completed", "failed"]


class PlanStep(TypedDict):
    """현재 turn의 계획에 포함된 개별 실행 단계."""

    id: int  # 계획 단계의 식별자. plan list의 index와 구분한다.
    task: str  # Executor가 수행할 독립적인 작업 지시.
    depends_on: list[int]  # 이 단계가 의존하는 선행 단계의 ID.
    status: PlanStepStatus
    result: str | None  # 이 단계의 결과 텍스트. 사용자 최종 응답과 구분한다.

    # 이 단계의 Executor 내부 대화와 Tool 호출·결과 메시지.
    # 대화 전체의 turns/current_turn과 별도이며, AnyMessage를 사용하지 않는다.
    messages: list[dict[str, Any]]

    # 이 단계에서 검색한 근거의 ID. 실제 근거는 TaskState에 보관한다.
    document_ids: list[str]
    memory_ids: list[str]


class TaskState(TypedDict):
    """현재 사용자 입력 한 건을 처리하기 위한 작업 상태."""

    goal: str  # 현재 turn에서 달성할 목표.
    plan: list[PlanStep]  # 재계획으로 대체된 단계도 이력으로 유지한다.
    current_plan_step_id: int | None  # 실행할 단계가 없으면 None.
    plan_revision: int  # 최초 0, 모델 재계획 또는 사용자 수정 시 증가한다.
    replan_count: int  # 모델 재계획 횟수. 사용자 직접 수정은 포함하지 않는다.

    # 검토 응답이 잘못되었을 때 다음 검토 요청에 표시할 오류.
    plan_review_error: str | None

    # ID를 key로 사용하여 근거를 중복 없이 보관한다.
    # 각 근거의 retrieved_by_plan_steps에 검색한 단계의 ID들을 보관한다.
    retrieved_documents: dict[str, dict[str, Any]]
    retrieved_memories: dict[str, dict[str, Any]]

    # 기존 코드의 실행 예산 계산에 사용하는 횟수.
    # 어떤 LLM 호출을 포함할지는 기존 count_llm_call 정책을 따른다.
    llm_call_count: int

    # 구조도에는 생략되었지만 원본의 실패 처리에서 사용하는 필드.
    aborted_reason: str | None
    trace: list[dict[str, Any]]


class AgentState(TypedDict):
    """여러 turn에 걸쳐 유지할 대화 상태와 현재 turn의 실행 상태."""

    # Conversation-level: 다음 turn으로 이어지는 대화 기록과 요약.
    turns: list[list[dict[str, Any]]]
    summary: str

    # Current-turn: 현재 사용자 입력 한 건과 그에 대한 처리 상태.
    # 실제 사용자·최종 assistant 메시지만 저장한다.
    current_turn: list[dict[str, Any]]
    task_state: TaskState | None

    # 마지막 Executor 호출 결과: {"status": ..., "content": ...}.
    # plan 안의 단계별 결과와 별도로, 다음 Node에 전달하는 실행 결과다.
    step_result: dict[str, str] | None

    # Progress Checker 결과: {"decision": ..., "reason": ...}.
    # decision은 기존 코드의 continue/replan/finish를 유지한다.
    progress: dict[str, str] | None

    final_response: str | None  # 현재 turn에 대한 사용자용 최종 응답.
    turn_status: TurnStatus


def create_plan_step(
    plan_step_id: int,
    task: str,
    depends_on: list[int],
) -> PlanStep:
    """기존 초기값으로 단계를 생성한다. 의존성 검증은 별도 함수가 담당한다."""
    return {
        "id": plan_step_id,
        "task": task,
        "depends_on": list(depends_on),
        "status": "pending",
        "result": None,
        "messages": [
            {"role": "user", "content": "Execute the current plan step."}
        ],
        "document_ids": [],
        "memory_ids": [],
    }


def create_task_state(goal: str) -> TaskState:
    """새 turn의 작업 상태를 생성한다. 기존 State는 변경하지 않는다."""
    return {
        "goal": goal,
        "plan": [],
        "current_plan_step_id": None,
        "plan_revision": 0,
        "replan_count": 0,
        "plan_review_error": None,
        "retrieved_documents": {},
        "retrieved_memories": {},
        "llm_call_count": 0,
        "aborted_reason": None,
        "trace": [],
    }


def create_state() -> AgentState:
    """대화 시작 전 초기 State를 생성한다. 새 turn마다 호출하지 않는다."""
    return {
        "turns": [],
        "summary": "",
        "current_turn": [],
        "task_state": None,
        "step_result": None,
        "progress": None,
        "final_response": None,
        "turn_status": "idle",
    }


def get_plan_step(task_state: TaskState, plan_step_id: int | None) -> PlanStep | None:
    return next(
        (step for step in task_state["plan"] if step["id"] == plan_step_id),
        None,
    )


def dependency_closure(task_state: TaskState, step: PlanStep) -> list[int]:
    """직접·간접 의존 단계를 계획 순서로 반환하고 잘못된 의존성을 검출한다."""
    required: set[int] = set()
    visiting: set[int] = set()

    def visit(plan_step_id: int) -> None:
        if plan_step_id in visiting:
            raise ValueError(f"Cyclic dependency at plan step {plan_step_id}.")
        if plan_step_id in required:
            return
        dependency = get_plan_step(task_state, plan_step_id)
        if dependency is None:
            raise ValueError(f"Unknown dependency plan step: {plan_step_id}")
        visiting.add(plan_step_id)
        for parent_id in dependency["depends_on"]:
            visit(parent_id)
        visiting.remove(plan_step_id)
        required.add(plan_step_id)

    for dependency_id in step["depends_on"]:
        visit(dependency_id)
    return [item["id"] for item in task_state["plan"] if item["id"] in required]


def artifacts_for_plan_steps(
    task_state: TaskState,
    plan_step_ids: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents = [
        artifact for artifact in task_state["retrieved_documents"].values()
        if plan_step_ids.intersection(artifact["retrieved_by_plan_steps"])
    ]
    memories_found = [
        artifact for artifact in task_state["retrieved_memories"].values()
        if plan_step_ids.intersection(artifact["retrieved_by_plan_steps"])
    ]
    return documents, memories_found


def compact_plan(plan: list[PlanStep]) -> list[dict[str, Any]]:
    """trace에 기록할 계획의 주요 필드만 추출한다."""
    return [
        {
            "id": step["id"],
            "task": step["task"],
            "depends_on": list(step["depends_on"]),
            "status": step["status"],
        }
        for step in plan
    ]


def next_ready_plan_step(plan: list[PlanStep]) -> PlanStep | None:
    """의존하는 단계가 모두 completed인 첫 pending 단계를 선택한다."""
    completed_ids = {step["id"] for step in plan if step["status"] == "completed"}
    return next(
        (
            step for step in plan
            if step["status"] == "pending"
            and set(step["depends_on"]).issubset(completed_ids)
        ),
        None,
    )


def turn_is_committed(task_state: TaskState) -> bool:
    """현재 task의 완료 trace로 대화 기록 확정 여부를 확인한다."""
    return any(
        item.get("event") == "turn_completed" for item in task_state["trace"]
    )

