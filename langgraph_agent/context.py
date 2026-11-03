"""각 LLM 단계에 필요한 정보만 골라 context와 영어 프롬프트를 구성한다."""
from __future__ import annotations

import json
from typing import Any
from config import MAX_PLAN_STEPS, RECENT_TURNS
from agent_state import (AgentState, PlanStep, TaskState, artifacts_for_plan_steps,
                         dependency_closure, get_plan_step)
from agent_state import compact_plan

MEMORY_POLICY = """
Memory Write Policy:

    - Remember newly provided or explicitly changed information from the current user message when it represents a durable requirement, preference, or decision that is likely to be useful in future conversations or tasks.
    - Call memory_write only when the current user message contains new information worth remembering.
    - When in doubt, do NOT call memory_write. A missed memory is cheaper than a polluted memory store.
    - The content passed to memory_write must contain only information newly obtained from the current user message.
    - Do not include information from previous turns, summaries, state, previous tool calls, or tool results in the content.
    - Do not combine or summarize multiple existing memories into a new memory.
    - Whenever possible, store only one independent requirement, preference, or decision in each memory.
    - Do not store information inferred, generated, or recommended by the Agent, unless the user explicitly adopts it as their own preference or decision.
    - Do not store greetings, casual conversation, temporary remarks, or information unlikely to be useful in future interactions. (e.g. weather, movies, thanks, jokes)


    Memory Retrieval Policy:

    - Call search_memory when the current context is insufficient and previously stored information is needed.
    - Do not call search_memory when the current context already contains enough information to answer or act correctly.
    - Generate a concise query that clearly describes the information that needs to be retrieved.
""".strip()


def select_completed_turns(
    state: AgentState,
    recent_turns: int | None,
) -> list[list[dict[str, Any]]]:
    """None이면 전체, 0이면 제외, 양수이면 최근 완료 turn을 선택한다."""
    if recent_turns is None:
        return state["turns"]
    if type(recent_turns) is not int or recent_turns < 0:
        raise ValueError("recent_turns must be None or a non-negative integer.")
    if recent_turns == 0:
        return []
    return state["turns"][-recent_turns:]


def turns_to_text(turns: list[list[dict[str, Any]]]) -> str:
    """요약 입력에는 실제 사용자·최종 assistant 텍스트만 포함한다."""
    lines = []
    for turn in turns:
        for message in turn:
            role = message.get("role")
            if role not in {"user", "assistant"} or message.get("tool_calls"):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_planner_context(
    state: AgentState,
    recent_turns: int | None = RECENT_TURNS,
    use_summary: bool = True,
) -> list[dict[str, Any]]:
    """기존 Planner 지시문, summary, 최근 완료 turn, 현재 turn으로 구성한다."""
    system_parts = [
        f"""
You are the Planner for an internal company-policy Agent.
Decompose the current user message into at most {MAX_PLAN_STEPS} executable
plan steps.

Executor capabilities (plan only within these):
- search internal company documents (policies, rules, procedures)
- read and write durable user information (memory)
- reason over evidence gathered by these tools
There is no web search, weather, calculator, or any other external tool.
If the goal cannot be served by these capabilities, plan a single step that
determines and states exactly what is unavailable.

Planning rules:
- Do not solve the goal and do not call tools.
- Each plan step must be self-contained because its Executor will not see the
  original user message, previous turns, summary, or the rest of the plan.
- Copy into a step all constraints that its Executor needs.
- A plan step is a meaningful subtask, not an individual tool call.
- Use depends_on to declare exactly which earlier step results are needed.
- Dependencies may refer only to earlier step IDs.
- Do not add a final-answer-writing step. Final response generation is separate.
- If durable new user information should be stored, create one self-contained
  step that quotes exactly that user-provided information.
- If prior user information is needed, create a self-contained memory-search step.

Return JSON only:
{{
  "steps": [
    {{"task": "self-contained subtask", "depends_on": []}},
    {{"task": "subtask using step 1", "depends_on": [1]}}
  ]
}}
""".strip()
    ]

    if use_summary and state["summary"]:
        system_parts.append("Earlier Conversation Summary:\n" + state["summary"])

    context = [{"role": "system", "content": "\n\n".join(system_parts)}]
    for turn in select_completed_turns(state, recent_turns):
        context.extend(turn)
    context.extend(state["current_turn"])
    return context


def build_executor_context(task_state: TaskState, step: PlanStep) -> list[dict[str, Any]]:
    """현재 단계와 의존 단계의 결과·근거만 LLM에 전달한다."""
    dependency_ids = dependency_closure(task_state, step)
    visible_step_ids = set(dependency_ids) | {step["id"]}
    documents, memories_found = artifacts_for_plan_steps(task_state, visible_step_ids)

    dependency_results = []
    for plan_step_id in dependency_ids:
        dependency = get_plan_step(task_state, plan_step_id)
        if dependency is None:
            raise ValueError(f"Unknown dependency plan step: {plan_step_id}")
        dependency_results.append({
            "id": dependency["id"],
            "task": dependency["task"],
            "result": dependency["result"],
        })

    payload = {
        "current_plan_step": {
            "id": step["id"],
            "task": step["task"],
            "depends_on": step["depends_on"],
        },
        "dependency_results": dependency_results,
        "relevant_documents": documents,
        "relevant_memories": memories_found,
    }

    system_content = f"""
You are the Executor for exactly one plan step.

Step Context:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Execution rules:
- Perform only current_plan_step.task.
- The step task's wording, including any examples it mentions, is an
  instruction only, never evidence. Base every policy claim solely on
  relevant_documents; if they do not support a claim, report it as missing.
- Do not infer or attempt the original user goal.
- Do not perform another plan step, even if it seems useful.
- Use only dependency_results and relevant evidence shown above.
- First inspect relevant_documents and answer from them when they contain
  sufficient evidence for the current plan step.
- Call search_documents only when relevant_documents are empty or insufficient.
- Do not repeat a document search whose results are already present in
  relevant_documents.
- Use tools when the current step requires them. One plan step may require
  multiple agent loop iterations and tool calls.
- Retrieved results are saved into state and will appear in Step Context on the
  next agent loop iteration.
- When this step is complete, respond without a tool call with a concise result.
- If required information cannot be found, state exactly what is missing.

{MEMORY_POLICY}
""".strip()

    return [
        {"role": "system", "content": system_content},
        *step["messages"],
    ]


def build_progress_context(
    task_state: TaskState,
    step: PlanStep,
    step_result: dict[str, str],
) -> list[dict[str, Any]]:
    """현재 단계·관련 근거·미실행 계획만 Checker에 전달한다."""
    remaining_steps = [
        {
            "id": item["id"],
            "task": item["task"],
            "depends_on": item["depends_on"],
        }
        for item in task_state["plan"]
        if item["status"] == "pending"
    ]
    visible_step_ids = set(dependency_closure(task_state, step)) | {step["id"]}
    documents, memories_found = artifacts_for_plan_steps(task_state, visible_step_ids)
    payload = {
        "current_plan_step": {
            "id": step["id"],
            "task": step["task"],
            "depends_on": step["depends_on"],
        },
        "executor_status": step_result["status"],
        "executor_result": step_result["content"],
        "relevant_documents": documents,
        "relevant_memories": memories_found,
        "remaining_steps": remaining_steps,
    }
    return [
        {
            "role": "system",
            "content": """
You are the Progress Checker. Do not solve the original user goal.

Choose exactly one decision:
- continue: the current step is sufficiently complete and remaining steps stay valid.
- replan: failure, missing evidence, or a new dependency invalidates the remaining plan.
- finish: no remaining step is needed after the current step.

Choose replan ONLY when a different action could plausibly obtain what is
missing. If the step already searched and the internal documents simply do
not contain the requested detail, that step is complete: choose continue or
finish, and the final answer will report the gap honestly. Never replan to
repeat a search that was already performed.

Return JSON only:
{"decision": "continue|replan|finish", "reason": "brief reason"}
""".strip(),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def build_replanner_context(
    task_state: TaskState,
    failed_step: PlanStep,
    reason: str,
) -> list[dict[str, Any]]:
    """목표·완료 결과·실패 관련 근거·남은 계획만 전달한다."""
    completed = [
        {"id": step["id"], "task": step["task"], "result": step["result"]}
        for step in task_state["plan"]
        if step["status"] == "completed"
    ]
    next_id = max((step["id"] for step in task_state["plan"]), default=0) + 1
    visible_step_ids = set(dependency_closure(task_state, failed_step)) | {
        failed_step["id"]
    }
    documents, memories_found = artifacts_for_plan_steps(task_state, visible_step_ids)
    payload = {
        "original_goal": task_state["goal"],
        "completed_steps": completed,
        "failed_step": {
            "id": failed_step["id"],
            "task": failed_step["task"],
            "result": failed_step["result"],
        },
        "reason_for_replanning": reason,
        "evidence_seen_by_failed_step": {
            "documents": documents,
            "memories": memories_found,
        },
        "old_remaining_plan": [
            step for step in compact_plan(task_state["plan"])
            if step["status"] == "pending"
        ],
        "next_available_plan_step_id": next_id,
    }
    return [
        {
            "role": "system",
            "content": f"""
You are the Replanner. Create only the remaining steps needed for the original
goal. Preserve completed work and do not repeat completed steps.

Rules:
- Each new step must be self-contained.
- Describe what each step must find or do, never what the answer might be.
  Do not put guessed policy content or illustrative examples into a task.
- Do not add a final-answer-writing step. If nothing remains to be done,
  return one step that states the conclusion reached so far.
- The first new step ID is {next_id}; use consecutive IDs.
- depends_on may refer to completed IDs or earlier new IDs only.
- If a new step needs evidence already gathered by a completed step, declare
  that completed step ID in depends_on instead of searching again.
- The executors have no web search or other external tools; do not plan
  steps that require capabilities beyond internal document search and memory.
- Return at most {MAX_PLAN_STEPS} new steps.

Return JSON only:
{{
  "remaining_steps": [
    {{"id": {next_id}, "task": "self-contained subtask", "depends_on": []}}
  ]
}}
""".strip(),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def build_final_context(
    state: AgentState,
    recent_turns: int | None = RECENT_TURNS,
    use_summary: bool = True,
) -> list[dict[str, Any]]:
    """완료 결과·근거·실패 정보와 실제 대화로 최종 응답용 context를 만든다."""
    task_state = state["task_state"]
    if task_state is None:
        raise RuntimeError("Finalizer requires task_state.")

    completed_results = [
        {"id": step["id"], "task": step["task"], "result": step["result"]}
        for step in task_state["plan"]
        if step["status"] == "completed"
    ]
    completed_ids = {
        step["id"] for step in task_state["plan"] if step["status"] == "completed"
    }
    documents, memories_found = artifacts_for_plan_steps(task_state, completed_ids)
    evidence: dict[str, Any] = {
        "completed_plan_step_results": completed_results,
        "retrieved_documents": documents,
        "retrieved_memories": memories_found,
    }
    failed_steps = [
        {"id": step["id"], "task": step["task"], "result": step["result"]}
        for step in task_state["plan"]
        if step["status"] == "failed"
    ]
    if failed_steps:
        evidence["failed_plan_steps"] = failed_steps
    if task_state["aborted_reason"]:
        evidence["task_aborted"] = task_state["aborted_reason"]

    system_parts = [
        f"""
You are an internal company-policy assistant. Produce the final answer to the
current user message using the completed task results below.

Task Results and Evidence:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Final-answer rules:
- Base company-policy claims only on retrieved internal documents.
- Prefer active documents and identify conflicts or missing evidence.
- Cite supporting internal sources by filename, year, and status.
- Use retrieved user memory only when relevant.
- If the evidence contains failed_plan_steps or task_aborted, state clearly
  what could not be determined and why; never invent the missing part.
- Do not mention the Planner, plan steps, progress checks, or task state.
""".strip()
    ]
    if use_summary and state["summary"]:
        system_parts.append("Earlier Conversation Summary:\n" + state["summary"])

    context = [{"role": "system", "content": "\n\n".join(system_parts)}]
    for turn in select_completed_turns(state, recent_turns):
        context.extend(turn)
    context.extend(state["current_turn"])
    return context


def build_summary_context(
    state: AgentState,
    recent_turns: int = RECENT_TURNS,
) -> list[dict[str, Any]] | None:
    """최근 turn과 중복되지 않는 이전 대화로 원본 요약 지시문을 구성한다."""
    if type(recent_turns) is not int or recent_turns < 0:
        raise ValueError("Summary recent_turns must be a non-negative integer.")
    if len(state["turns"]) <= recent_turns:
        return None

    older_turns = state["turns"] if recent_turns == 0 else state["turns"][:-recent_turns]
    transcript = turns_to_text(older_turns)
    if not transcript:
        return None
    return [
        {
            "role": "system",
            "content": (
                "Summarize only important constraints, user preferences, and "
                "decisions that may be needed later. Preserve numerical and "
                "mandatory requirements. Omit plans, intermediate reasoning, "
                "tool calls, tool results, and temporary task details."
            ),
        },
        {"role": "user", "content": transcript},
    ]

