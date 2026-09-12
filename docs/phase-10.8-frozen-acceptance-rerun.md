# Phase 10.8: Frozen Live Acceptance Rerun

## Purpose

Phases 10.4–10.7 changed the contextual router prompt roughly ten times and were
verified only with targeted `gpt-5.6-luna` probes. Neither frozen live fixture had
been re-run since it was recorded. This rerun checks that earlier accepted
behavior still holds.

Both runs used `gpt-5.6-terra`, the model pinned by the Phase 10 fixture and used
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

**Not changed in this phase.** The fixture is frozen evidence, and rewriting its
expectation to match new behavior would erase exactly the signal a frozen fixture
exists to give. The options are recorded for a decision:

- Keep the fixture as is and record this case as a known divergence caused by
  `2b0c199`.
- Revise the fixture so the case is ambiguous on its own terms — for example, two
  candidate meetings — rather than through a date the tool used to reject.
- Separately, treat an update that changes nothing as a no-op reply instead of an
  audited `Updated` operation.

### Run 3 — `runs/phase10-live-20260912-c`, commit `8e8052e`

Re-run after `fix: choose tracker candidates by relevance to the message`, which
changes candidate selection on every turn. The result is identical to run 2: 15
of 16, with only `ambiguous_correction_effect` failing for the reason above. The
candidate change introduced no regression.

## Phase 8 tracker acceptance

`runs/phase8-live-20260912`, commit `7ff17e1`: all 12 assertions passed with no
errors — task creation and completion, decision rationale, review and
supersession, blocker resolution, deletion, the casual no-change turn, status and
history sources, project isolation, and prepared-operation resume. This matches
the recorded 2026-09-08 result, so the Phase 10.6–10.7 tracker changes
(`due_at`, `set_deadline`) did not regress the Phase 6–7 services.

## Deterministic suite

252 tests pass at `7ff17e1`, and 254 at `8e8052e`.

## Claim boundary

Each run is one fixed-sequence demonstration with one model. They show that the
accepted behavior is intact apart from the case explained above; they are not a
reliability measurement.
