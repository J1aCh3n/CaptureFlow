# Phase 10.6 Update: Tracker Deadlines on the Calendar

## Purpose

Close the limitation recorded in Phase 10.5: a stated deadline became a tracker
task that was invisible in Calendar. A deadline must be visible on the calendar
without becoming an event.

Baseline at diagnosis:

- Branch: `codex/contextual-router-phase10`
- Commit: `dc39fd0` (`docs: record the phase 10.5 compound-turn decisions`)
- Local timezone: `America/Vancouver`
- Reference date during the observed runs: `2026-09-11`

## The design question

`CalendarEvent` was built for meetings: it is start-anchored, with `start_at`
and an optional `end_at`. A deadline is the opposite shape — humans read it as
an end instant with no start.

Three options were weighed:

1. A synthetic event running from `deadline - 15 minutes` to the deadline. It
   needs no schema change and draws a block on a calendar grid, but it fabricates
   a start time nobody stated. That directly contradicts the standing rule
   "never invent a default duration" (Phase 10.3 exit gate 5), and the fabricated
   value would enter the audit snapshots that exist precisely so every stored
   field traces back to the user's words.
2. A real `CalendarEvent` marked as a deadline, with `end_at` left null. No
   fabricated data, but the deadline then exists twice — once as the tracker task
   and once as the event — with no way to keep them in step.
3. A structured deadline on the tracker record, projected read-only into the
   Calendar view.

Two facts settled it. First, `ProjectRecord` had no due field at all, so
`下周三晚上八点` survived only as free text inside the task content and no view
could sort or render it. Second, the tracker record is already the thing the user
completes, so any second copy would immediately drift.

**Decision (taken by the user): the deadline is displayed on the calendar but is
not an event and occupies no time.** That is option 3.

## Change 1: a structured deadline on the task

`ProjectChange` gains three fields:

```python
due_date_expression: str | None   # copied from the message by the model
due_time_expression: str | None   # copied from the message by the model
due_at: date | datetime | None    # resolved by Python, never by the model
```

Rules, all enforced in Python:

- Only a `create_record` change whose kind is `task` may carry any of them; a
  transition or supersession may not.
- A deadline time requires a deadline date.
- The router refuses any decision in which the model supplied `due_at` itself.
  The model only copies expressions; `resolve_relative_date()` and `parse_time()`
  resolve them, exactly as they do for Calendar.
- An expression Python cannot resolve does not become a guessed timestamp. The
  task is still created, its deadline fields are cleared, and the reply says the
  deadline was not understood.
- A date with no stated clock time stays a plain date, not midnight.

`ProjectRecord` gains `due_at`, which only a task may carry. Migration `013` adds
the column and an index; it is a plain additive `ALTER TABLE`, with no table
rebuild. `list_task_deadlines()` returns active dated tasks in due order.

Commit: `feat: store a structured deadline on tracker tasks`

## Change 2: the read-only Calendar projection

The Calendar page renders a **Deadlines** block above the events. Each entry
shows the due instant, the project name, and the task text. It has no edit,
delete, or undo control, and the caption states that deadlines are not events and
reserve no time. Completing or deleting the task in Tracker removes it from
Calendar, because nothing was copied: the tracker record remains the only source
of truth.

The project snapshot's **Upcoming** list merges deadlines with events in date
order, labelling a deadline as such.

This change also fixes `save_project_record()`, whose insert statement dropped
`due_at` while the project-change path stored it.

Commit: `feat: show tracker deadlines on the calendar without creating events`

## Deterministic tests

- A tracker task with `下周三` and `晚上八点` resolves to
  `2026-09-16T20:00:00-07:00`, appears in `list_task_deadlines()`, and disappears
  from it once the task is completed.
- A deadline with a date but no time stays the plain date `2026-09-16`.
- An unresolvable deadline keeps the task, sets no `due_at`, and says so.
- A model-supplied `due_at` is refused and writes no record.
- Schema tests reject deadline fields on a non-task kind, on a transition, and a
  deadline time with no deadline date.
- A UI test asserts the Calendar page shows the deadline, creates no
  `CalendarEvent`, exposes no event delete control for it, and drops it after the
  task is completed.

The full local suite passes 230 tests.

## Live check

Run against `gpt-5.6-luna` on 2026-09-11 in a project-assigned conversation, one
model call per turn:

```text
turn 1 (the reported compound message)
  -> intent tracker_change, model_calls 1
  -> task record with due_at 2026-09-16T20:00:00-07:00
  -> reply states the deadline and asks about the meeting minutes
  -> list_task_deadlines() returns that one task

turn 2 (是的，请保存。)
  -> intent save_note, model_calls 1
  -> minutes stored, pending question resolved, deadline unchanged
```

Bounded demonstration evidence only. Raw model output and credentials are not
retained.

## Explicit non-goals

- No synthetic duration, and no default event length for a deadline.
- No second stored copy of a deadline in `calendar_events`.
- No reminders, notifications, or overdue escalation.
- No deadline on a blocker, question, decision, or constraint yet; only tasks
  carry one.
