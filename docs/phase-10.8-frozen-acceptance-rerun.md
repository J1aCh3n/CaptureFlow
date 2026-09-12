# Phase 10.8: Frozen Live Acceptance Rerun

## Purpose

Phases 10.4–10.7 changed the contextual router prompt roughly ten times and were
verified only with targeted `gpt-5.6-luna` probes. Neither frozen live fixture had
been re-run since it was recorded. This rerun checks that earlier accepted
behavior still holds.

Every run used `gpt-5.6-terra`, the model pinned by the Phase 10 fixture and used
for the recorded Phase 8 evidence, so the results are comparable. Traces are in
the ignored `runs/` directory; no raw model output or credentials are copied here.

## Phase 10 contextual router

### Run 1 — `runs/phase10-live-20260912`, commit `052d64f`

Failed at the first case, and every later case that depended on the event
cascaded.

| Case | Result |
| --- | --- |
| `calendar_create` | `clarification_created` instead of an event |
| `calendar_update` | crashed with `sqlite3.IntegrityError` |
| downstream effects | failed for lack of the event |

Two independent defects, both fixed before the next run:

1. **A bare range end was unresolved.** For
   `我明天下午四点到五点有一场算法优化与代码审查会议` the model copied the end
   time literally as `五点` — exactly what the prompt asks for — and a bare hour
   does not parse. The recorded 2026-09-09 run passed only because the model then
   happened to write `下午五点`. A bare end now takes the period stated on its
   start, but only when that gives an end after the start.
   Commit: `fix: let a bare range end share the start's period`.
2. **A second clarification crashed the turn.** Only one clarification may be
   pending per conversation. When a message needed a Calendar clarification while
   an earlier one was unanswered, the insert violated the unique index, and no
   router handler catches `IntegrityError`. This predates Phase 10.4; the cascade
   above exposed it. The newer question now replaces the unanswered one.
   Commit: `fix: replace an unanswered clarification instead of crashing`.

### Run 2 — `runs/phase10-live-20260912-b`, commit `7ff17e1`

15 of 16 records passed. Creation, correction to the new time range, task
distinction, meeting question, replay with zero model calls, and both
cross-conversation lookups all match the recorded evidence.

The one failure is `ambiguous_correction_effect`.

### Why `ambiguous_correction` fails, and why this rerun did not cause it

The case sends `这个会议改到周四。` from a second conversation, with reference
date 2026-09-09, a Wednesday, and expects a pending clarification.

At the recorded evidence commit `803b2a3`, `date_tool` classified every bare
weekday such as `周四` as `ambiguous`, which is what forced that clarification. On
2026-09-10, before this rerun series began, commit `2b0c199`
(`feat: resolve local timezone and future weekdays`) deliberately made a bare
weekday resolve to its nearest future occurrence. `周四` now resolves to
2026-09-10, the model proposes `calendar_update`, and it commits. The frozen
fixture was not re-run after that change, so its expectation has been stale since
then.

What the run actually did is also worth noting: the event was already on
Thursday 2026-09-10, so the committed update changed nothing, yet it wrote an
audited operation and replied `Updated local calendar event`.

**The recorded pass was a workaround, not a fix.** The baseline contract
(`docs/phase-10-baseline.md`) says a correction without a sufficient target must
clarify, and this case is sent from a second conversation in which `这个会议` has
no antecedent at all. It passed because a date was rejected, never because the
target was checked. The fixture itself was therefore left unchanged; rewriting a
frozen expectation to match new behavior would erase the signal it exists to give.

**Resolution.** The user chose to enforce the target in Python rather than to hand
the model extra hint fields in its candidate payload. The model still chooses the
target; Python only refuses to write an update or delete that the current
conversation does not support — no message in the bounded window created or
changed the event, and the current message does not name it with a distinctive
term. Such a turn now asks which event is meant and records `target` as the
reason. Commit: `fix: ask which event when a conversation never identified it`.

The no-op the run exposed was fixed separately: an update that leaves the event
unchanged writes nothing and is reported as `calendar_unchanged`.
Commit: `fix: do not audit an update that changes nothing`.

### Run 3 — `runs/phase10-live-20260912-c`, commit `8e8052e`

Re-run after `fix: choose tracker candidates by relevance to the message`, which
changes candidate selection on every turn. The result is identical to run 2: 15
of 16, with only `ambiguous_correction_effect` failing for the reason above. The
candidate change introduced no regression.

### Run 4 — `runs/phase10-live-20260912-d`, commit `28ff5b8`

With the target rule in place, `ambiguous_correction` passed for the right reason:
the model still proposed `calendar_update`, and Python turned it into a `target`
clarification. But `calendar_update` in the main conversation, which had passed
twice, now failed. For `把这场会议改到明天下午两点到三点` the model split the text
into date `明天` and time `两点`, leaving `下午` in neither field, so the bare hour
was unresolved. Two fixes followed:

- `fix: restore a period the model split away from its hour` — when a copied hour
  does not parse and the message writes a period directly before it, reattach it.
- `fix: only a write of the pending kind answers a question` — the same run showed
  that logging an unrelated tracker task had marked the pending calendar time
  question resolved, silently discarding the event being created. That behavior
  was introduced in Phase 10.5; now only a write of the pending kind resolves it.

### Run 5 — `runs/phase10-live-20260912-e`, commit `6e0dcc7`

**16 of 16 passed, with the frozen fixture unchanged.** `ambiguous_correction`
shows the intended split of responsibility: intent `calendar_update` from the
model, outcome `clarification_created` with reason `target` from Python.

Across these five runs the same model split the same inputs three different ways.
One fully passing run shows the rules now absorb the variants observed; it is not
a measure of how often a new variant will appear.

## Phase 8 tracker acceptance

`runs/phase8-live-20260912`, commit `7ff17e1`: all 12 assertions passed with no
errors — task creation and completion, decision rationale, review and
supersession, blocker resolution, deletion, the casual no-change turn, status and
history sources, project isolation, and prepared-operation resume. This matches
the recorded 2026-09-08 result, so the Phase 10.6–10.7 tracker changes
(`due_at`, `set_deadline`) did not regress the Phase 6–7 services.

## Deterministic suite

252 tests pass at `7ff17e1`, 254 at `8e8052e`, and 262 at `6e0dcc7`.

## Claim boundary

Each run is one fixed-sequence demonstration with one model. They show that the
accepted behavior is intact apart from the case explained above; they are not a
reliability measurement.

Phase 8 was not re-run after `7ff17e1`: every later change is in the contextual
router, candidate selection, or time parsing, none of which the Phase 6–7
`ProjectTrackerService` and `ProjectRetrievalService` paths exercised by that
fixture use.
