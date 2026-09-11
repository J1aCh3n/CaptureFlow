# Phase 10.5 Update: Compound Turns and Finished Meetings

## Purpose

Resolve the compound-turn question left open by Phase 10.3 and restated in
Phase 10.4. One message that states two different local items must no longer
discard one of them silently.

Baseline at diagnosis:

- Branch: `codex/contextual-router-phase10`
- Commit: `ea53cc4` (`feat: delete local calendar events from chat`)
- Local timezone: `America/Vancouver`
- Reference date during the observed runs: `2026-09-11`

## Reported case

```text
今天中午十二点的会议是关于系统架构与驱动集成，主要分析了ALOHA 遥操作系统的驱动调试。
排查 ROS 2 发布者-订阅者节点之间的延迟问题，并测试 MoveIt 2 的运动规划框架。
修复需要下周三晚上八点之前提交
```

The user's intent was two records: the meeting minutes, and the stated deadline.
After the Phase 10.4 clock fix the turn routed to one `calendar_create` for the
already-finished meeting and dropped the deadline entirely.

## Product decisions taken

Both were decided by the user during this feedback round:

1. **One action per turn stays.** A turn commits at most one local action. When
   the message states a second item, CaptureFlow asks about it rather than
   committing it or discarding it.
2. **Route a meeting by its tense.** A meeting reported as already held keeps
   its minutes as a Note; only a meeting that is still going to happen becomes a
   Calendar event.

## Change 1: tense-based meeting routing

The contextual prompt now distinguishes a meeting that is still going to happen
from one reported as already held or finished. The first keeps `calendar_create`;
the second uses `save_note` and creates no Calendar event. Tense is judged from
the message's own wording relative to `reference_date`.

Commit: `feat: keep a finished meeting as a note, not a calendar event`

## Change 2: a deferred second item

`ConversationDecision` gains two paired fields:

```python
deferred_action: ContextualAction | None = None
follow_up_question: str | None = Field(default=None, max_length=300)
```

Rules enforced in Python, not by the model:

- Both fields are set together or not at all.
- Only a decision that commits one local action may defer a second item.
- A deferred action may only be `create_event`, `save_idea`, or `save_note`. It
  may not update or delete an existing record, so a deferred item can never
  silently change stored data.
- The deferred action must cite the current message, like every other action.
- Because a tracker change cannot be deferred, a message that states both a
  tracker item and a Calendar/idea/note item commits the tracker item and defers
  the other one.

When a turn commits its action and carries a deferred item, the router appends
the follow-up question to the reply and persists the deferred action as a
pending clarification with the new `missing_field` value `follow_up`. Migration
`012` widens the `pending_clarifications` CHECK constraint to include it; the
table is rebuilt with the same additive pattern used by migration `011`.

The next turn receives that pending action as clarification context. A
confirmation executes it through ordinary validation — evidence, candidates, and
date/time resolution are unchanged — and any committed write now resolves the
open question, which previously only happened on the Calendar branches.

The Chat page shows the follow-up question alone in its pending banner and
offers **Dismiss suggestion** instead of **Cancel pending event**.

Commit: `feat: ask about a stated second item instead of dropping it`

## Supporting fix: evidence on a confirmation turn

Every local write must cite the current user message. A confirmation such as
`是的，请保存。` contains none of the committed content, and the first live run
failed closed with `a local write must cite the current user message`.

The prompt now states that one evidence entry must always quote the current
message exactly, even when that message only confirms a pending item, and that
the earlier message is cited as a second entry when the content comes from it.
The Python rule is unchanged.

## Deterministic tests

- A compound turn saves the minutes, asks about the stated deadline, persists a
  `follow_up` clarification holding `下周三 / 晚上八点`, and a confirming turn then
  creates the event at `2026-09-16T20:00:00-07:00` and resolves the question.
- A committed tracker deadline defers the meeting minutes; the confirmation saves
  exactly one Note and clears the pending question.
- A deferred item on a decision that commits nothing is refused, with no pending
  clarification and no Calendar write.
- Schema tests cover the paired fields and reject a deferred `delete_event`.

The full local suite passes 224 tests.

## Live check

Run against `gpt-5.6-luna` on 2026-09-11 in a project-assigned conversation,
one model call per turn, throwaway local SQLite:

```text
turn 1 (the reported compound message)
  -> intent tracker_change, outcome tracker_change, model_calls 1
  -> one active task record for the deadline
  -> pending follow_up clarification holding a save_note action
  -> no Calendar event, no Note yet

turn 2 (是的，请保存。)
  -> intent save_note, outcome save_note, model_calls 1
  -> one Note stored, pending clarification resolved
```

An earlier run of the same message chose `save_note` first and deferred the
deadline as a Note, which lost the deadline's kind. That is why committing the
tracker item is now an explicit prompt rule rather than a free choice.

This is bounded demonstration evidence. Raw model output and credentials are not
retained.

## Known limitation

A deadline with an explicit date and time becomes a tracker task, not a Calendar
event, because the established contract routes a user-owned deadline to the
tracker. A user who expects `下周三晚上八点` to appear in Calendar will not find
it there. Whether a dated deadline should also create a Calendar event is a
separate product decision and is not implemented here.

Resolved in [phase-10.6-tracker-deadlines-on-the-calendar.md](phase-10.6-tracker-deadlines-on-the-calendar.md):
the deadline stays a tracker task and is displayed read-only on the Calendar page
without becoming an event.

Declining a follow-up question is also not a routed intent yet: an unanswered
question stays pending until the user answers it or presses **Dismiss
suggestion**.
