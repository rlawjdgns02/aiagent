"""Graph State 정의, State 생성 함수, plan 조회·검증 함수.

AgentState
├── turns          완료된 턴들의 실제 대화        (대화 수준: 턴이 끝나도 남는다)
├── summary        오래된 턴의 요약               (대화 수준)
├── current_turn   이번 턴의 실제 대화            (턴 수준: 턴마다 새로 시작)
└── current_task   TaskState                      (턴 수준: 턴마다 새로 시작)
    ├── plan       list[PlanStep]
    └── ...
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from contexts import MAX_PLAN_STEPS


# OpenAI chat 형식의 메시지: {"role": ..., "content": ..., "tool_calls"?: ..., "tool_call_id"?: ...}
Message = dict[str, Any]


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class PlanStep(TypedDict):
    id: int
    instruction: str
    depends_on: list[int]
    status: Literal["pending", "in_progress", "completed", "failed", "replaced"]
    result: str | None

    # Internal tool-call protocol messages for this plan step only.
    messages: list[Message]

    # IDs of artifacts produced by this plan step.
    document_ids: list[str]
    memory_ids: list[str]


class StepResult(TypedDict):
    status: Literal["completed", "failed"]
    content: str


class ProgressResult(TypedDict):
    decision: Literal["continue", "replan", "finish"]
    reason: str


class TaskState(TypedDict):
    goal: str
    plan: list[PlanStep]
    current_plan_step_id: int | None
    plan_revision: int
    replan_count: int

    # Full task-level evidence is retained here with plan-step provenance.
    retrieved_documents: dict[str, dict[str, Any]]
    retrieved_memories: dict[str, dict[str, Any]]

    llm_call_count: int
    aborted_reason: str | None
    trace: list[dict[str, Any]]

    # 노드 사이에 넘기는 실행 정보. interrupt 후 재개해도 남아 있어야 하므로 State에 둔다.
    turn_status: Literal["completed", "failed"]
    step_result: StepResult | None        # executor → 라우터, progress_checker, replanner
    progress: ProgressResult | None       # progress_checker → 라우터, replanner
    review_error: str | None              # plan_review의 잘못된 수정 사유 → 다시 검토 요청


class AgentState(TypedDict):
    # 누적이 필요한 유일한 키: final_answer가 [이번 턴]을 반환하면 기존 목록 뒤에 붙는다.
    turns: Annotated[list[list[Message]], operator.add]
    summary: str
    current_turn: list[Message]
    current_task: TaskState | None


# ---------------------------------------------------------------------------
# State constructors
# ---------------------------------------------------------------------------

def create_state() -> AgentState:
    """새 대화(thread)의 첫 State."""

    return {
        # Conversation state: survives across completed turns.
        "turns": [],
        "summary": "",

        # Current turn/task state: replaced when the next turn starts.
        "current_turn": [],
        "current_task": None,
    }


def create_task_state(goal: str) -> TaskState:
    return {
        "goal": goal,
        "plan": [],
        "current_plan_step_id": None,
        "plan_revision": 0,
        "replan_count": 0,

        "retrieved_documents": {},
        "retrieved_memories": {},

        "llm_call_count": 0,
        "aborted_reason": None,
        "trace": [],

        "turn_status": "completed",
        "step_result": None,
        "progress": None,
        "review_error": None,
    }


def create_plan_step(
    plan_step_id: int,
    instruction: str,
    depends_on: list[int],
) -> PlanStep:
    return {
        "id": plan_step_id,
        "instruction": instruction,
        "depends_on": depends_on,
        "status": "pending",
        "result": None,
        "messages": [
            {
                "role": "user",
                "content": "Execute the current plan step.",
            }
        ],
        "document_ids": [],
        "memory_ids": [],
    }


def trace_event(event: str, **details: Any) -> dict[str, Any]:
    """trace 한 줄을 만든다."""

    return {"event": event, **details}


# ---------------------------------------------------------------------------
# Plan helpers (입력 plan을 수정하지 않는다)
# ---------------------------------------------------------------------------

def get_plan_step(plan: list[PlanStep], plan_step_id: int | None) -> PlanStep | None:
    for step in plan:
        if step["id"] == plan_step_id:
            return step
    return None


def next_pending_plan_step(plan: list[PlanStep]) -> PlanStep | None:
    for step in plan:
        if step["status"] == "pending":
            return step
    return None


def update_plan_step(
    plan: list[PlanStep],
    plan_step_id: int,
    **changes: Any,
) -> list[PlanStep]:
    """해당 step만 바뀐 새 plan 리스트를 만든다. 다른 step은 그대로 재사용한다."""

    return [
        {**step, **changes} if step["id"] == plan_step_id else step
        for step in plan
    ]


def compact_plan(plan: list[PlanStep]) -> list[dict[str, Any]]:
    return [
        {
            "id": step["id"],
            "instruction": step["instruction"],
            "depends_on": step["depends_on"],
            "status": step["status"],
        }
        for step in plan
    ]


# ---------------------------------------------------------------------------
# Plan validation (Planner 출력과 사용자 수정 계획에 모두 쓴다)
# ---------------------------------------------------------------------------

def normalize_initial_plan(raw_steps: Any) -> list[PlanStep]:
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
    plan: list[PlanStep],
) -> list[PlanStep]:
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
