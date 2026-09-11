# Phase 10.4 Update: Conversational Calendar Deletion and Noon Times

## Purpose

Record the first hands-on user-feedback round after Phase 10.3. Two defects were
reported from ordinary use. Both are fixed here. A third, larger routing question
that the same session exposed is stated but deliberately not implemented.

Baseline at diagnosis:

- Branch: `codex/contextual-router-phase10`
- Commit: `504f39f` (`feat: support explicit calendar durations`)
- Local timezone: `America/Vancouver`
- Reference date during the observed run: `2026-09-11`

## Reported case 1: a noon meeting asked for a start time

Input:

```text
今天中午十二点的会议是关于系统架构与驱动集成，主要分析了ALOHA 遥操作系统的驱动调试。
排查 ROS 2 发布者-订阅者节点之间的延迟问题，并测试 MoveIt 2 的运动规划框架。
修复需要下周三晚上八点之前提交
```

Observed reply:

```text
What exact start time should I use for this local event?
```

### Verified root cause

`src/captureflow/time_tool.py` accepted only `上午`, `下午`, and `晚上`. `中午`
was outside the bounded vocabulary, so `中午十二点` was unresolved and reached
the Phase 10.3 fail-closed time clarification.

The same hour mapping also guessed silently at the twelve-o'clock boundary.
Direct checks at the baseline commit produced:

```text
中午十二点 -> None
晚上十二点 -> 12:00
上午十二点 -> 00:00
下午十二点 -> 12:00
```

`晚上十二点` naming 12:00 noon is wrong, and every twelve-o'clock phrase other
than `中午十二点` is ambiguous about which calendar day it belongs to.

### Change

`_parse_chinese_time()` now accepts `凌晨` and `中午`, and a new `_period_hour()`
resolves the period:

- `中午十二点` resolves to 12:00; `中午` with any other hour is unresolved.
- `凌晨` and `上午` keep hours 1–11 as written.
- `下午` and `晚上` add twelve to hours 1–11.
- A twelve-o'clock hour under `凌晨`, `上午`, `下午`, or `晚上` is unresolved and
  reaches the existing clarification instead of a guessed timestamp.

Commit: `fix: resolve 中午 clock times and reject ambiguous 12点`

## Reported case 2: an expired meeting could not be deleted from chat

The user created a meeting in a project conversation, then asked in a later turn:

```text
昨天的会议已结束，删除过期会议
```

Chat could not remove the event. Deletion existed only as the Calendar page's
per-event **Delete event** button; the contextual router had no delete intent at
all, so no conversational request could reach it.

### Change

`calendar_delete` is now a routed intent:

- `ContextualAction` accepts a `delete_event` action. It carries only
  `target_event_id` and evidence: content, description, and any date, time, or
  duration expression are rejected.
- `ConversationDecision` requires a candidate target for `calendar_delete`, and
  `clarify` may carry a pending `delete_event` action with no target.
- `ConversationRouterService._delete_event()` writes the deletion through the
  existing `calendar_event_update_operations` audit table. The snapshot pair is
  the active event before and the same event with `status = "deleted"` after.
- `CalendarEventUpdateOperation` allows `after.status` to be `deleted` while
  `before.status` must be `active`, and rejects a deletion that also edits
  `title`, `description`, `start_at`, `end_at`, `all_day`, or `timezone`.
- The Chat card labels the operation as a deletion and offers **Undo calendar
  deletion**; Undo restores the exact pre-deletion snapshot.

Reusing the update audit table means preparation, commit, replay idempotency,
restart resume, workspace scoping, and evidence provenance are inherited rather
than reimplemented. No migration was required.

Commit: `feat: delete local calendar events from chat`

## Deterministic tests

Added to `tests/test_time_tool.py`:

```text
中午十二点 -> 12:00
中午12点半 -> 12:30
凌晨三点   -> 03:00
晚上十二点 / 上午十二点 / 凌晨十二点 / 中午三点 -> unresolved
```

Added to `tests/test_models.py`:

- `calendar_delete` requires a candidate target; a bare `delete_event` is valid
  only as a clarification's pending action.
- A `delete_event` action carrying a time expression is rejected.
- A delete operation whose `after` snapshot also renames the event is rejected.

Added to `tests/test_conversation_router_service.py`:

- The reported delete turn commits exactly one audited operation, marks the
  event deleted, replays with zero model calls, and restores the exact original
  event on Undo.
- An ambiguous delete creates a pending `delete_event` clarification, leaves
  both events active, and writes no operation.
- A delete naming an event outside the supplied candidates fails closed and
  changes nothing.

The full local suite passes 218 tests.

## Live check

Run against `gpt-5.6-luna` with the configured endpoint on 2026-09-11, one model
call per turn, using a throwaway local SQLite file:

```text
single delete   -> intent calendar_delete, outcome calendar_delete, model_calls 1,
                   target event status deleted, no pending clarification
ambiguous delete -> intent clarify, outcome clarification_created, model_calls 1,
                   both events still active, pending action delete_event with a
                   null target
```

This is bounded demonstration evidence for the new intent, not a reliability
benchmark. Raw model output and credentials are not retained.

## Open question, not implemented here

The case 1 message is a compound turn. It states a past meeting and its minutes,
plus a separate deadline (`下周三晚上八点`). After the parser fix it routes to
one `calendar_create` for the past meeting at 12:00 today and drops the deadline
entirely; the user's intent was to record meeting minutes and to schedule the
deadline.

Phase 10.3 already flagged the same behavior and deferred it as a product
decision. It stays deferred here: one message still produces at most one local
action. Resolving it requires deciding whether one turn may commit more than one
bounded local action, and whether a stated past meeting should become a Note
rather than a Calendar event.
