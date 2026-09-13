# Phase 12: What `local_answer` and `project_answer` Mean

**Status:** implemented. The user's decisions are recorded below.

## Problem

Both intents answer a question without writing anything, and in practice they
overlapped:

- **`local_answer` ignored the model's choice.** The model listed the candidate it
  meant in `source_ids`, but the router discarded the list and ran its own date
  and keyword lookup.
- **`project_answer` saw almost nothing of the project.** The model wrote its reply
  from at most three active records chosen by keyword overlap. It could not see
  progress, all open work, deadlines, milestones, or a replaced decision, so it
  could not answer a question about the project as a whole.
- **A workaround papered over the gap.** Because `project_answer` text is the
  model's own, `e3bff8d` sent any `project_answer` that cited a Calendar event to
  the local lookup, so the time would come from SQLite. That silently re-routed a
  misclassified turn instead of making each intent right.

## Definitions

| Intent | Question | Who writes the answer | What it sees |
| --- | --- | --- | --- |
| `local_answer` | About one or a few specific stored items: a Calendar event, an idea or note, or a single tracker record such as a task's deadline | Python, from the SQLite rows | The candidates the router already supplies |
| `project_answer` | About the project as a whole: progress, outstanding work, blockers, deadlines across tasks, or why a decision changed | A second model call, from a project snapshot Python assembles | The whole bounded project snapshot |

## Decisions

1. `project_answer` is written by a **second model call** made only for that intent.
   Every other turn still makes one call.
2. A question about **a single task** is `local_answer`.
3. The snapshot **includes history**: superseded decisions and constraints, their
   rationale, and the record that replaced each one.
4. `e3bff8d` is **reverted**. Each intent is made correct instead.
5. The router prompt changes to state these definitions. The routing schema
   does not change.

## `local_answer`

- The model lists the candidates the question is about in `source_ids`: events,
  ideas/notes, or project records.
- Python renders exactly those rows. Times and deadlines come from SQLite, and
  source cards are saved for them. The model's reply text is not shown.
- If `source_ids` is empty, Python falls back to the Phase 9 date and keyword
  lookup, because the item may not be among the three candidates the router supplied.
- A `reply` is no longer required for either answer intent. Requiring it was
  another structural rule the decision did not need.

## `project_answer`

1. **The routing call only classifies.** Its `reply` and `source_ids` are unused.
2. **Python builds the snapshot from SQLite** for the conversation's project, with a
   bound on each part:
   - active records of every kind, with deadline and rationale (up to 40);
   - recently completed tasks and milestones, and resolved questions and blockers
     (up to 15);
   - superseded decisions and constraints, each linked to its replacement (up to 15);
   - the project's upcoming Calendar events (up to 5).
3. **The second call** receives the question, the recent messages, the snapshot, and
   the reference date. It returns `reply` and `source_ids`, drawn from the
   snapshot's records and events.
4. **Python checks every cited ID** against the snapshot. An ID outside it fails the
   turn, because the reply may rest on a fact that is not stored.
5. **Python renders dates.** Under the reply it lists each cited task with its
   deadline and each cited event with its time, taken from SQLite. The prompt tells
   the model not to state dates or times. This is not enforced by pattern matching,
   which proved brittle in Phase 9.1; a wrong date in the prose would sit next to
   the stored one.
6. **Source cards** are saved for cited records and events, up to six each.
7. **Failures:**
   - If the conversation has no project, the turn replies that it cannot answer a
     project question.
   - If the second call fails, the turn reports `model_or_validation_failure` with
     its reason and two model calls.

The second call's schema tolerates structural noise the way the routing schema now
does: a `null` source list is empty, duplicate IDs are collapsed, and unknown keys
are ignored.

## Acceptance

`meeting_question` and `cross_conversation_local_retrieval` check the effect: the
local lookup answered, nothing was written, and the created event is among the
source cards. `cross_conversation_project_retrieval` records two model calls.
The protocol version changes with these checks.

## Evidence

The deterministic suite passes 177 tests.

A live probe on `qwen3.8-flash` before committing, in a seeded project, answered
all three questions correctly:
- "这个项目现在进展如何？还卡在哪里？" became `project_answer` with two calls,
  summarized the completed and open work and the blocker, cited six records, and
  listed the stored deadline under the reply.
- "为什么后来不用 SQLite 了？" was answered from the superseded decision and its
  replacement's rationale.
- "排查延迟那个任务什么时候截止？" became `local_answer`, and the deadline was rendered
  from SQLite.

Protocol version 2 ran five times on `qwen3.8-flash` at `b203a86`, from a clean
working tree, with eight model calls per run:

| Case | Passed |
| --- | --- |
| `calendar_create` | 5 of 5 |
| `calendar_update` | 5 of 5 |
| `tracker_change` | 4 of 5 |
| `meeting_question` | 5 of 5 |
| `replay_idempotency` | 5 of 5 |
| `cross_conversation_local_retrieval` | 5 of 5 |
| `cross_conversation_project_retrieval` | 4 of 5 |
| `ambiguous_correction` | 3 of 5 |

Records per run were 14, 15, 16, 15, and 16 of 16; runs are
`runs/phase12-v2-1` to `-5`. Model time per run was 151–379 seconds.

- **`tracker_change` failed once.** The decision put an `evidence` list at its top
  level, which the schema forbids; a tracker change quotes its source inside each
  change. That run's project answer then had no tracked task to cite.
- **`ambiguous_correction` failed twice**, for the two reasons found when that turn
  was replayed: once evidence cited an assistant reply, and once a `clarify` came
  without a pending action.
- **The answer cases held.** Both meeting questions passed every run. The project
  question failed only in run 1, where there was no task to cite.

## Not in scope

- How `ambiguous_correction` should be defined, and whether the target rule runs
  before evidence checks. Both are still open.
- Streaming, caching, or shrinking the snapshot for very large projects beyond the
  bounds above.
