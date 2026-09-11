# Phase 10.3 Update: Chinese Time and Explicit Duration

## Purpose

Fix the remaining Calendar-update failure exposed by the Phase 10.2 manual
acceptance test. Keep the change bounded to local Calendar time parsing,
explicit duration handling, clarification behavior, and regression evidence.

Baseline branch and commit at diagnosis:

- Branch: `codex/contextual-router-phase10`
- Commit: `2b0c199` (`feat: resolve local timezone and future weekdays`)
- Local timezone: `America/Vancouver`
- Reference date during the observed run: `2026-09-10`

Do not treat this document as evidence that the implementation is complete.
The exit gates below must be executed after the code changes.

## Reproduction

First turn:

```text
今天开始进行A测，下午三点钟要准备好，四点有个相关的说明会
```

Observed result:

```text
Created local calendar event: 相关的说明会 — 2026-09-10T16:00:00-07:00 (America/Vancouver).
```

Persisted Calendar state:

```text
title: 相关的说明会
description: 今天开始进行A测，下午三点钟要准备好
start_at: 2026-09-10T16:00:00-07:00
end_at: null
timezone: America/Vancouver
status: active
```

Second turn:

```text
会议延期了，时间也被缩短到半个小时，现在晚上七点才开始
```

Observed result:

```text
I could not safely apply that local change: event start time is not valid
```

Verified failure effect:

- No Calendar update operation was created.
- The existing event remained active at 16:00.
- `end_at` remained `null`.
- The failure was fail-closed; it did not partially modify the event.

## Important interpretation

`2026-09-10T16:00:00-07:00` is an ISO timestamp. `-07:00` is the UTC offset
for Vancouver daylight time; it is not an event duration or end time.

The original turn did not state an event duration or end time. Keeping
`end_at = null` is therefore correct under the existing rule that CaptureFlow
must not invent a default duration.

The second turn does state an explicit duration: `半个小时`. The intended
updated interval is therefore:

```text
start_at: 2026-09-10T19:00:00-07:00
end_at:   2026-09-10T19:30:00-07:00
```

## Verified root causes

### 1. Contextual time parsing regressed behind the legacy parser

`src/captureflow/conversation_router_service.py::_parse_time()` currently
accepts colon-based values and a small hard-coded dictionary:

```text
noon
midnight
下午两点
下午三点
下午四点
下午五点
```

Direct checks at the baseline commit produced:

```text
下午四点       -> 16:00
晚上七点       -> None
晚上七点才开始 -> None
19:00          -> 19:00
19:30          -> 19:30
下午六点       -> None
```

The legacy `src/captureflow/router_service.py` already has broader Chinese
clock parsing for `上午`, `下午`, and `晚上`, Arabic or Chinese hour numerals,
and half hours. Phase 10 introduced a second, narrower implementation instead
of reusing the established parser.

### 2. The contextual action contract cannot represent a duration

`ContextualAction` currently has:

```text
time_expression
end_time_expression
```

It has no exact `duration_expression`. Asking the model to convert `半个小时`
into `19:30` would move arithmetic and normalization into the model and weaken
the deterministic evidence boundary. The duration should be copied from the
message and resolved by Python.

### 3. Unsupported time reaches an internal-style failure reply

The dispatcher tries to apply the update and raises `event start time is not
valid`. A user-understandable time expression that is outside the bounded
parser should create a clarification and no write, not expose this validation
message as the final assistant response.

## Required behavior

For the two-turn reproduction above, the second turn must:

1. select the existing `相关的说明会` event;
2. preserve its date, title, description, timezone, and provenance;
3. resolve `晚上七点` to 19:00 local time;
4. resolve `半个小时` to 30 minutes;
5. calculate `end_at` deterministically as 19:30;
6. commit exactly one audited Calendar update;
7. allow the update to be undone;
8. use one model proposal call for the normal turn;
9. avoid creating a duplicate event or Tracker record.

## Recommended implementation

### 1. Use one shared local-time parser

Create a small shared module such as `src/captureflow/time_tool.py`, or move the
existing robust legacy implementation there. Both `router_service.py` and
`conversation_router_service.py` should import the same parser.

Keep the parser deterministic. At minimum support:

- `19:00` and full-width-colon `19：00`;
- `7 PM`, `7:30 PM`, `noon`, and `midnight`;
- `上午九点`, `上午九点半`;
- `下午六点`;
- `晚上七点`, `晚上七点半`;
- Arabic and bounded Chinese hour numerals.

The model should copy the smallest exact time expression, such as `晚上七点`,
instead of including surrounding words such as `才开始`.

Do not maintain two independent time parsers after this change.

### 2. Add an explicit duration expression

Extend `ContextualAction` with:

```python
duration_expression: str | None = None
```

Add a deterministic duration resolver. The initial bounded vocabulary should
cover only common, unambiguous forms:

- `半小时`, `半个小时`;
- `30分钟`, `45分钟`, `60分钟`;
- `一小时`, `1小时`, `2小时`;
- equivalent simple English minute/hour forms if they are already within the
  product's bilingual scope.

Use a positive bounded duration. Reject zero, negative, malformed, or
unreasonably long values. Do not silently guess a duration when none is stated.

Rules:

- `end_time_expression` and `duration_expression` must not both be populated.
- On create, a duration requires a valid start time.
- On update, a new start plus duration yields `end_at = start_at + duration`.
- On update, duration alone applies to the existing start time.
- Explicit arithmetic belongs to Python, not the model.
- Crossing midnight should use normal datetime arithmetic rather than being
  rejected merely because the resulting end date is the next day.

### 3. Clarify unsupported time or duration safely

Generalize the existing Calendar date-clarification path so unsupported or
ambiguous time/duration input can produce a pending clarification with no
write. If `PendingClarification.missing_field` remains typed as date-only,
extend it narrowly to the required bounded values.

Expected user-facing replies should resemble:

```text
What exact start time should I use for this local event?
```

or:

```text
How long should this local event last (for example, 30 minutes)?
```

Do not return `event start time is not valid` for ordinary supported-language
input.

### 4. Preserve audited provenance and idempotency

- Keep all `EvidenceReference` rows from the original event and correction.
- Preserve prepared-to-committed restart behavior.
- Retrying the same message ID must not create a second update.
- Undo must restore the exact pre-update start and end values.
- A validation or parsing failure must create no partial operation and make no
  Calendar change.

## Required deterministic tests

### Shared time parser

Add a parameterized test matrix for:

```text
下午四点       -> 16:00
下午六点       -> 18:00
晚上七点       -> 19:00
晚上七点半     -> 19:30
上午九点半     -> 09:30
19:00          -> 19:00
19：30         -> 19:30
7 PM           -> 19:00
7:30 PM        -> 19:30
```

Also test malformed and ambiguous values. They must return a typed unresolved
result or `None`; they must not raise an uncontrolled exception.

### Duration resolver

Cover at least:

```text
半个小时 -> 30 minutes
半小时   -> 30 minutes
30分钟   -> 30 minutes
一小时   -> 60 minutes
2小时    -> 120 minutes
0分钟    -> rejected
-30分钟  -> rejected
```

### Contextual Calendar update

Freeze the reported sequence as a scripted service test. Assert:

```text
intent: calendar_update
outcome: calendar_update
model_calls: 1
event count: unchanged
start_at: 2026-09-10T19:00:00-07:00
end_at: 2026-09-10T19:30:00-07:00
update operation count: 1
pending clarification: none
```

Then exercise Undo and assert the event returns to:

```text
start_at: 2026-09-10T16:00:00-07:00
end_at: null
```

Add separate tests for:

- unsupported start time creates clarification and no update;
- unsupported duration creates clarification and no update;
- explicit `19:00` to `19:30` still works;
- create without duration keeps `end_at = null`;
- replay with the same source message ID performs zero extra model calls and
  creates no duplicate update;
- prepared update resumes after reopening the database.

### Prompt and schema

Verify that the contextual prompt:

- asks the model to copy an explicit duration rather than calculate an end;
- asks for the smallest exact date/time/duration expression;
- does not permit a default duration;
- keeps `end_time_expression` and `duration_expression` mutually exclusive.

## Live acceptance case

Do not make a live API call without explicit cost approval. After deterministic
gates pass, append one frozen sequence using the exact reported Chinese inputs.

Expected live result:

```text
turn 1: calendar_create, one event at 16:00, end_at null
turn 2: calendar_update, same event at 19:00-19:30
```

Record only redacted intent, outcome, model-call count, timing, stable IDs, and
final local state. Do not retain raw model output or credentials.

## Explicit non-goals

Do not expand this update into:

- external Calendar synchronization;
- recurring-event support;
- email, Zoom, or notification integration;
- general natural-language temporal reasoning;
- a default event duration;
- arbitrary multi-action execution from one message.

The first reproduction message also contains `下午三点钟要准备好`. Phase 10.2
kept that sentence in the meeting description rather than creating a separate
Tracker task or reminder. That compound-turn behavior is worth a separate
product decision, but it is not part of this time/duration repair.

## Exit gates

This update is complete only when all of the following are true:

1. The exact reported two-turn sequence passes deterministically.
2. `晚上七点` and `半个小时` are resolved by Python, not model arithmetic.
3. The updated event is 19:00-19:30 in `America/Vancouver`.
4. Failed parsing produces clarification and zero writes.
5. Creation without a stated duration still keeps `end_at = null`.
6. Provenance, replay, restart, and Undo tests pass.
7. The complete existing deterministic suite passes.
8. The prior Phase 10 live fixture remains unchanged and passes if a live rerun
   is separately approved.
9. `compileall` and `git diff --check` pass.

Before running the app or tests from this worktree, verify that Python imports
this checkout rather than the editable installation of the main checkout:

```powershell
$env:PYTHONPATH = (Resolve-Path '.\src').Path
& 'C:\Users\frank\anaconda3\envs\llm-course\python.exe' -c "import captureflow; print(captureflow.__file__)"
```

The printed path must contain `captureflow-contextual-router\src\captureflow`.

