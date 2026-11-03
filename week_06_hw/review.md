# Week 06 Review — Raw steps → Normalized plan

Planner LLM이 만든 `raw_steps`가 [`normalize_initial_plan`](agent_loop.py#L247)을 거쳐 실행 가능한 plan step으로 바뀌는 과정

**예시 질문:** 2026년 기준 해외 출장 숙박비·식비 한도와 출발 전 승인 요건은?

---

## 1. Raw steps (LLM이 생성한 plan)

LLM은 `instruction`과 `depends_on`만 생성. step 번호는 리스트 순서(1부터)로 암묵적으로 정함.

```python
[
    {
        'instruction': '해외 출장 관련 정책 문서에서 2026 년 기준 숙박비와 식비의 한도 금액을 검색한다.',
        'depends_on': [],
    },
    {
        'instruction': '해외 출장 관련 정책 문서에서 출발 전 승인 요건 (예: 사전 승인서 제출, 일정 승인 등) 을 검색한다.',
        'depends_on': [],
    },
    {
        'instruction': '검색된 숙박비, 식비 한도 및 출발 전 승인 요건 정보를 종합하여 정리한다.',
        'depends_on': [1, 2],
    },
]
```

의존 관계:

```text
[1] 숙박비·식비 한도 검색 ─┐
                           ├──▶ [3] 종합 정리
[2] 출발 전 승인 요건 검색 ─┘
```

---

## 2. Normalized plan

[`create_plan_step`](agent_loop.py#L79)이 각 step에 실행용 필드를 붙임.

```python
[
    {
        'id': 1,
        'instruction': '해외 출장 관련 정책 문서에서 2026 년 기준 숙박비와 식비의 한도 금액을 검색한다.',
        'depends_on': [],
        'status': 'pending',
        'result': None,
        'messages': [{'role': 'user', 'content': 'Execute the current plan step.'}],
        'document_ids': [],
        'memory_ids': [],
    },
    {
        'id': 2,
        'instruction': '해외 출장 관련 정책 문서에서 출발 전 승인 요건 (예: 사전 승인서 제출, 일정 승인 등) 을 검색한다.',
        'depends_on': [],
        'status': 'pending',
        'result': None,
        'messages': [{'role': 'user', 'content': 'Execute the current plan step.'}],
        'document_ids': [],
        'memory_ids': [],
    },
    {
        'id': 3,
        'instruction': '검색된 숙박비, 식비 한도 및 출발 전 승인 요건 정보를 종합하여 정리한다.',
        'depends_on': [1, 2],
        'status': 'pending',
        'result': None,
        'messages': [{'role': 'user', 'content': 'Execute the current plan step.'}],
        'document_ids': [],
        'memory_ids': [],
    },
]
```

---

## 3. 무엇이 바뀌었나

| 필드 | Raw | Normalized | 의미 |
|---|:---:|:---:|---|
| `id` | – | ✅ | 리스트 순서(1부터)를 명시적 ID로 부여 |
| `instruction` | ✅ | ✅ | `strip()` 후 빈 문자열이면 에러 |
| `depends_on` | ✅ | ✅ | 이전 step만 참조 가능(`1 <= dep < id`), 중복 제거·정렬 |
| `status` | – | ✅ | `'pending'`으로 시작 → 실행하며 갱신 |
| `result` | – | ✅ | step 실행 결과 (초기 `None`) |
| `messages` | – | ✅ | 이 step 전용 tool-call 대화 기록 (시작 메시지 1개) |
| `document_ids` | – | ✅ | 이 step이 만든 문서 artifact ID |
| `memory_ids` | – | ✅ | 이 step이 만든 memory ID |

### 검증 규칙 (실패 시 `ValueError`)

- `raw_steps`가 비어 있지 않은 list여야 함
- step 수가 `MAX_PLAN_STEPS` 이하
- 각 step이 dict
- `instruction`이 비어 있지 않음
- `depends_on`의 모든 값이 자기보다 앞선 step 번호 → 순환·전방 참조 차단

> **핵심:** LLM은 *무엇을 할지*(instruction, depends_on)만 정하고, 실행 상태(status, result, messages, ids)는 코드가 일관된 형태로 채운다.
