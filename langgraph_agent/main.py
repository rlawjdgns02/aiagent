"""터미널에서 질문, 계획 검토, 대화와 검증 결과 저장을 담당한다."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from langgraph.types import Command

from agent_graph import build_agent_graph
from agent_state import create_state
from config import DEBUG, MODEL_NAME, OUTPUT_DIR, RECENT_TURNS, USE_SUMMARY
from llm import create_client
from tools import TOOLS, TOOL_FUNCTIONS


class SessionStopped(Exception):
    """사용자가 현재 대화를 종료했다."""


def save_snapshot(graph: Any, config: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """현재 State와 trace를 JSON으로 각각 저장한다. API 키는 State에 없다."""
    state = dict(graph.get_state(config).values)
    output_dir.mkdir(parents=True, exist_ok=True)
    thread_id = str(config["configurable"]["thread_id"])
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", thread_id) or "session"
    state_path = output_dir / f"state_{safe_id}.json"
    trace_path = output_dir / f"trace_{safe_id}.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    trace = (state.get("task_state") or {}).get("trace", [])
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return state_path, trace_path


def read_plan_review(
    payload: dict[str, Any],
    output_dir: Path,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], Any] = print,
) -> dict[str, Any]:
    """승인하거나, 편집용 JSON 파일에서 수정된 pending 계획 전체를 읽는다."""
    output_fn("\n[계획 검토]")
    output_fn(json.dumps(payload, ensure_ascii=False, indent=2))
    while True:
        action = input_fn("Enter/approve=승인, edit=수정, quit=종료 > ").strip().lower()
        if action in {"", "a", "approve", "승인"}:
            return {"action": "approve"}
        if action in {"q", "quit", "exit", "종료"}:
            raise SessionStopped()
        if action not in {"e", "edit", "수정"}:
            output_fn("approve, edit, quit 중 하나를 입력하세요.")
            continue

        editable_ids = set(payload["editable_step_ids"])
        draft = [
            {"id": step["id"], "task": step["task"], "depends_on": step["depends_on"]}
            for step in payload["plan"] if step["id"] in editable_ids
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        edit_path = output_dir / f"plan_review_{uuid4().hex[:8]}.json"
        edit_path.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
        output_fn(f"다음 JSON 파일에서 task와 depends_on을 수정하세요: {edit_path}")
        output_fn("미실행 단계 전체를 남기세요. 완료 이력의 ID는 수정할 수 없습니다.")
        while True:
            response = input_fn("수정 후 Enter=제출, cancel=승인/수정 선택으로 돌아가기 > ").strip().lower()
            if response in {"cancel", "취소"}:
                break
            try:
                edited = json.loads(edit_path.read_text(encoding="utf-8-sig"))
                return {"action": "edit", "edited_plan": edited}
            except (OSError, json.JSONDecodeError) as error:
                output_fn(f"JSON 파일을 읽을 수 없습니다: {error}")


def drive_graph(
    graph: Any,
    payload: Any,
    config: dict[str, Any],
    output_dir: Path,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], Any] = print,
) -> dict[str, Any]:
    """검토 중단과 Node 오류를 처리하며 현재 turn의 종료까지 진행한다."""
    while True:
        try:
            result = graph.invoke(payload, config)
        except Exception as error:
            state_path, _ = save_snapshot(graph, config, output_dir)
            snapshot = graph.get_state(config)
            output_fn(f"\n실행 오류: {type(error).__name__}: {error}")
            output_fn(f"재시도할 Node: {snapshot.next}; 현재 State: {state_path}")
            if snapshot.values.get("final_response"):
                output_fn("이미 생성된 최종 응답:\n" + snapshot.values["final_response"])
            while True:
                answer = input_fn("retry=현재 Node 재시도, quit=종료 > ").strip().lower()
                if answer in {"retry", "r", "재시도"}:
                    payload = None
                    break
                if answer in {"quit", "q", "exit", "종료"}:
                    raise SessionStopped() from error
            continue

        if "__interrupt__" in result:
            save_snapshot(graph, config, output_dir)
            review = read_plan_review(result["__interrupt__"][0].value, output_dir, input_fn, output_fn)
            payload = Command(resume=review)
            continue

        state_path, trace_path = save_snapshot(graph, config, output_dir)
        task = result["task_state"]
        output_fn(f"\n[{result['turn_status']} / 실행 LLM 호출 {task['llm_call_count']}회 / 재계획 {task['replan_count']}회]")
        output_fn(result["final_response"])
        output_fn(f"State: {state_path}\nTrace: {trace_path}")
        return result


def run_question(
    graph: Any,
    question: str,
    config: dict[str, Any],
    output_dir: Path,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], Any] = print,
) -> dict[str, Any]:
    """첫 turn에만 초기값을 전달하고 이후에는 새 current_turn만 전달한다."""
    if not question.strip():
        raise ValueError("질문을 입력하세요.")
    snapshot = graph.get_state(config)
    if snapshot.next:
        raise RuntimeError("새 질문 전에 중단된 turn을 재개해야 합니다.")
    payload = {"current_turn": [{"role": "user", "content": question}]}
    if not snapshot.values:
        payload = {**create_state(), **payload}
    return drive_graph(graph, payload, config, output_dir, input_fn, output_fn)


def main() -> int:
    parser = argparse.ArgumentParser(description="계획 승인·수정을 지원하는 LangGraph Agent")
    parser.add_argument("--thread-id", default="session_1")
    parser.add_argument("--question", help="질문 한 건을 처리한 뒤 종료한다. 계획 검토 입력은 받는다.")
    parser.add_argument("--debug", action="store_true", help="단계별 context를 출력한다.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    try:
        client = create_client()
        model = MODEL_NAME
        if not model:
            available = client.models.list().data
            if not available:
                raise RuntimeError("서버에서 사용할 수 있는 모델이 없습니다.")
            model = available[0].id
        graph = build_agent_graph(
            client=client, model=model, tool_schemas=TOOLS, tool_functions=TOOL_FUNCTIONS,
            recent_turns=RECENT_TURNS, use_summary=USE_SUMMARY, debug=args.debug or DEBUG,
        )
        config = {"configurable": {"thread_id": args.thread_id}}
        print(f"모델: {model}; thread_id: {args.thread_id}")
        print("종료: /quit. 이 프로세스 안에서 Graph와 대화 checkpoint를 재사용합니다.")
        if args.question:
            run_question(graph, args.question, config, args.output_dir)
            return 0
        while True:
            question = input("\n사용자 > ").strip()
            if question.lower() in {"/quit", "/exit"}:
                return 0
            if question:
                run_question(graph, question, config, args.output_dir)
    except (SessionStopped, EOFError, KeyboardInterrupt):
        print("\n대화를 종료합니다.")
        return 0
    except Exception as error:
        print(f"설정 또는 실행 오류: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

