"""Phase-specific context builders for the planning Agent.

The full Agent state is retained by agent_loop.py. This module only projects
the subset that each LLM phase is allowed to see. Context builders do not call
the LLM and do not mutate state.
"""

from __future__ import annotations

import json
from typing import Any


# Shared with plan validation in agent_loop.py.
MAX_PLAN_STEPS = 8


# ---------------------------------------------------------------------------
# Context selection helpers
# ---------------------------------------------------------------------------

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
    state: dict[str, Any],
    recent_turns: int | None,
) -> list[list[dict[str, Any]]]:
    if recent_turns is None:
        return state["turns"]
    return state["turns"][-recent_turns:]


def _get_plan_step(
    state: dict[str, Any],
    plan_step_id: int,
) -> dict[str, Any] | None:
    """Look up a plan step without importing the orchestration module."""

    for step in state["current_task"]["plan"]:
        if step["id"] == plan_step_id:
            return step
    return None


def _compact_plan(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only the plan fields needed in an LLM context."""

    return [
        {
            "id": step["id"],
            "instruction": step["instruction"],
            "depends_on": step["depends_on"],
            "status": step["status"],
        }
        for step in state["current_task"]["plan"]
    ]


def dependency_closure(
    state: dict[str, Any],
    step: dict[str, Any],
) -> list[int]:
    """Return transitive dependency IDs in plan order."""

    required: set[int] = set()

    def visit(plan_step_id: int) -> None:
        if plan_step_id in required:
            return
        dependency = _get_plan_step(state, plan_step_id)
        if dependency is None:
            raise ValueError(f"Unknown dependency plan step: {plan_step_id}")
        for parent_id in dependency["depends_on"]:
            visit(parent_id)
        required.add(plan_step_id)

    for dependency_id in step["depends_on"]:
        visit(dependency_id)

    plan_order = [item["id"] for item in state["current_task"]["plan"]]
    return [plan_step_id for plan_step_id in plan_order if plan_step_id in required]


def artifacts_for_plan_steps(
    state: dict[str, Any],
    plan_step_ids: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents = [
        artifact
        for artifact in state["current_task"]["retrieved_documents"].values()
        if plan_step_ids.intersection(artifact["retrieved_by_plan_steps"])
    ]
    memories_found = [
        artifact
        for artifact in state["current_task"]["retrieved_memories"].values()
        if plan_step_ids.intersection(artifact["retrieved_by_plan_steps"])
    ]
    return documents, memories_found


def build_planner_context(
    state: dict[str, Any],
    current_turn: list[dict[str, Any]],
    recent_turns: int | None,
    use_summary: bool,
) -> list[dict[str, Any]]:
    """Planner sees conversation context and the new user goal."""

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
    {{"instruction": "self-contained subtask", "depends_on": []}},
    {{"instruction": "subtask using step 1", "depends_on": [1]}}
  ]
}}
""".strip()
    ]

    if use_summary and state["summary"]:
        system_parts.append("Earlier Conversation Summary:\n" + state["summary"])

    context = [{"role": "system", "content": "\n\n".join(system_parts)}]
    for turn in select_completed_turns(state, recent_turns):
        context.extend(turn)
    context.extend(current_turn)
    return context


def build_executor_context(
    state: dict[str, Any],
    step: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build a context isolated to one plan step.

    Deliberately excluded:
    - original goal/current_turn
    - full plan and unrelated pending steps
    - previous conversation turns and summary
    - evidence from unrelated plan steps
    """

    dependency_ids = dependency_closure(state, step)
    visible_step_ids = set(dependency_ids) | {step["id"]}
    documents, memories_found = artifacts_for_plan_steps(state, visible_step_ids)
    
    dependency_results = []
    for plan_step_id in dependency_ids:
        dependency = _get_plan_step(state, plan_step_id)
        dependency_results.append(
            {
                "id": dependency["id"],
                "instruction": dependency["instruction"],
                "result": dependency["result"],
            }
        )

    payload = {
        "current_plan_step": {
            "id": step["id"],
            "instruction": step["instruction"],
            "depends_on": step["depends_on"], ## 제거해도 된다
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
- Perform only current_plan_step.instruction.
- The step instruction's wording, including any examples it mentions, is an
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
    state: dict[str, Any],
    step: dict[str, Any],
    step_result: dict[str, str],
) -> list[dict[str, Any]]:
    """Progress Checker sees the current step and only its forward boundary."""

    remaining_steps = [
        {
            "id": item["id"],
            "instruction": item["instruction"],
            "depends_on": item["depends_on"],
        }
        for item in state["current_task"]["plan"]
        if item["status"] == "pending"
    ]
    visible_step_ids = set(dependency_closure(state, step)) | {step["id"]}
    documents, memories_found = artifacts_for_plan_steps(state, visible_step_ids)
    payload = {
        "current_plan_step": {
            "id": step["id"],
            "instruction": step["instruction"],
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
    state: dict[str, Any],
    failed_step: dict[str, Any],
    reason: str,
) -> list[dict[str, Any]]:
    """Replanner needs global task structure, but not raw conversation history."""

    current_task = state["current_task"]
    completed = [
        {
            "id": step["id"],
            "instruction": step["instruction"],
            "result": step["result"],
        }
        for step in current_task["plan"]
        if step["status"] == "completed"
    ]
    next_id = max((step["id"] for step in current_task["plan"]), default=0) + 1
    visible_step_ids = set(dependency_closure(state, failed_step)) | {
        failed_step["id"]
    }
    documents, memories_found = artifacts_for_plan_steps(state, visible_step_ids)
    payload = {
        "original_goal": current_task["goal"],
        "completed_steps": completed,
        "failed_step": {
            "id": failed_step["id"],
            "instruction": failed_step["instruction"],
            "result": failed_step["result"],
        },
        "reason_for_replanning": reason,
        "evidence_seen_by_failed_step": {
            "documents": documents,
            "memories": memories_found,
        },
        "old_remaining_plan": [
            step for step in _compact_plan(state) if step["status"] == "pending"
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
  Do not put guessed policy content or illustrative examples into an instruction.
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
    {{"id": {next_id}, "instruction": "self-contained subtask", "depends_on": []}}
  ]
}}
""".strip(),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def build_final_context(
    state: dict[str, Any],
    current_turn: list[dict[str, Any]],
    recent_turns: int | None,
    use_summary: bool,
) -> list[dict[str, Any]]:
    """Final Answer phase may integrate the entire completed task."""

    current_task = state["current_task"]
    completed_results = [
        {
            "id": step["id"],
            "instruction": step["instruction"],
            "result": step["result"],
        }
        for step in current_task["plan"]
        if step["status"] == "completed"
    ]
    completed_ids = {
        step["id"]
        for step in current_task["plan"]
        if step["status"] == "completed"
    }
    documents, memories_found = artifacts_for_plan_steps(state, completed_ids)
    evidence = {
        "completed_plan_step_results": completed_results,
        "retrieved_documents": documents,
        "retrieved_memories": memories_found,
    }

    # 실패한 턴에서도 최종 답변은 생성한다. 대신 무엇이 왜 안 됐는지를
    # 증거로 넘겨, 모델이 빈 부분을 지어내지 않고 설명하게 한다.
    failed_steps = [
        {
            "id": step["id"],
            "instruction": step["instruction"],
            "result": step["result"],
        }
        for step in current_task["plan"]
        if step["status"] == "failed"
    ]
    if failed_steps:
        evidence["failed_plan_steps"] = failed_steps
    if current_task["aborted_reason"]:
        evidence["task_aborted"] = current_task["aborted_reason"]

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
    context.extend(current_turn)
    return context


# ---------------------------------------------------------------------------
# Summary: completed turns, not plan steps, are the summarization unit
# ---------------------------------------------------------------------------

def turns_to_text(turns: list[list[dict[str, Any]]]) -> str:
    lines = []
    for turn in turns:
        for message in turn:
            role = message["role"]
            if role == "assistant" and message.get("tool_calls"):
                names = [call["function"]["name"] for call in message["tool_calls"]]
                lines.append("assistant used tools: " + ", ".join(names))
                continue
            if role == "tool":
                continue
            if message.get("content"):
                lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)


def build_summary_context(
    state: dict[str, Any],
    recent_turns: int,
) -> list[dict[str, Any]] | None:
    if len(state["turns"]) <= recent_turns:
        return None

    transcript = turns_to_text(state["turns"][:-recent_turns])
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



