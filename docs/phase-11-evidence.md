# Phase 11 Evidence: Model-Normalized Dates and Times

Design: [phase-11-model-normalized-time-design.md](phase-11-model-normalized-time-design.md).

## What was implemented

| Commit | Change |
| --- | --- |
| `acf9996` | Calendar reference table passed to the model on every turn |
| `361361d` | Calendar actions carry `date`, `start_time`, `end_time`, `duration_minutes`, `timezone`, `recurring`; Python attaches the timezone and applies policy; update replies state the new schedule |
| `fff0306` | Tracker deadlines carry `due_date`, `due_time`, `recurring`; stored pre-Phase-11 proposals still load |
| `e7f6d7c` | Fix found live: a pending update may lack the date it is asking about |
| `ea90f31` | Fix found live: a repeating event may lack a date and still reach the recurrence policy |

The contextual router no longer imports any parsing from `time_tool` or
`date_tool`; it uses only `calendar_reference`.

## Deterministic evidence

- 267 tests pass at `ea90f31`.
- Behavioral assertions from Phases 10.4–10.8 were kept and their scripted
  decisions rewritten to the normalized fields. Tests that existed only for Python
  wording rules — restoring a split period, inheriting a range end's period — were
  removed, because those rules no longer exist.
- New tests cover the `-08:00` offset on 2026-11-02, the update echo-back,
  malformed dates and times, an implausible date, the calendar table, a pending
  update without a date, a repeating event without a date, and a stored
  pre-Phase-11 deadline change.

## Compatibility with stored data

A copy of the local database was upgraded and checked. Migrations 14–16 applied.
Both committed tracker operations that store the retired `due_date_expression`
key validate — their normalized fields are derived from the stored `due_at`
(`2026-09-12 10:00` and `2026-09-16 20:00`) — and both undo successfully.

## Frozen Phase 10 live acceptance (`gpt-5.6-terra`)

| Run | Commit | Result |
| --- | --- | --- |
| `runs/phase11-phase10-live-20260912` | `fff0306` | 15 of 16 |
| `runs/phase11-phase10-live-20260912-b` | `e7f6d7c` | **16 of 16** |

The frozen fixture was not changed.

The first run's only failure was `ambiguous_correction`, with a validation
failure rather than a clarification. It was reproduced from that run's database
state: in one attempt out of four, the model did exactly what the new prompt asks
— it could not tell which Thursday `这个会议改到周四` meant and chose `clarify`
with a sensible question, 9月10日 or 9月17日, leaving the pending update's date
null. The schema required every update to carry a replacement field, so a correct
clarification was rejected. Under the old contract the model copied `周四` into
`date_expression`, which satisfied that rule by accident. The requirement now
applies only to a committed `calendar_update`.

In the other three attempts the model proposed a `calendar_update` for
2026-09-10, and the Phase 10.8 target rule turned it into a clarification.

## `gpt-5.6-luna` boundary probes

Reference date 2026-09-11, one model call per turn, a fresh database per case.
The Phase 10.7 column is the behavior recorded there, before later fixes.

| Input | Phase 10.7 | Phase 11 |
| --- | --- | --- |
| `9月20日上午十点有体检` | asked for a date | event 2026-09-20 10:00 |
| `明天下午两点开评审会，开一个半小时` | asked for a duration | 14:00–15:30 |
| `Team sync tomorrow at 2 PM for an hour.` | asked for a duration | 14:00–15:00 |
| `下下周三下午三点有架构评审会` | asked for a date | event 2026-09-23 15:00 |
| `月底之前要把报告交上去` | task, deadline not resolved | task, deadline 2026-09-30 |
| `纽约时间明天上午九点有电话会` | stored 09:00 local, silently | `timezone` clarification |
| `每周一下午三点有团队例会` | asked for a date | `recurrence` clarification |
| `提醒我明天十点吃药` | task, deadline dropped | task, deadline 2026-09-12 10:00 |
| `我明天下午四点到五点有一场…会议` | failed in Phase 10.8 | 16:00–17:00 |
| `把这场会议改到明天下午两点到三点` | failed in Phase 10.8 | 14:00–15:00, schedule echoed |
| `明天晚饭后十点跟家里人视频` | — | 22:00 |
| `11月2日下午三点有季度评审` | — | 15:00 `-08:00` |
| `把站会提前一小时` | asked for a start time | 10:00 moved to 09:00 |
| `上周五开了个会，讨论了驱动集成的排期` | note | note |
| `周一上午十点有站会，周二下午三点有需求评审` | first created, second asked | first created, second asked |
| `我下周有什么安排？` | `insufficient_context` | `insufficient_context` |

`每周一…` failed validation three times out of three on first probe: the model
represented a weekly schedule faithfully, with `recurring: true`, a start time, and
no date, and the rule that a start time needs a date rejected it before the
recurrence policy could answer. That rule now skips a recurring action, which is
never committed.

## Observations

- **The accepted ambiguity trade-off is visible.** `提醒我明天十点吃药` became a
  10:00 deadline without a question. The reply states the time, so it can be seen
  and undone; this is the behavior decision 1 accepted.
- **Both live defects had the same shape.** A structural rule ran ahead of the
  policy meant to handle the case. Under the old contract, copied words happened
  to satisfy those rules; normalized `null` values do not. Other structural rules
  that assume a field is present on an action that is not yet committed are worth
  checking with that in mind.
- **The target rule is stricter than natural reading.** In the acceptance replay,
  the later conversation had asked about `算法优化与代码审查会议` by name two
  turns earlier, yet `这个会议改到周四` still clarified, because only messages that
  created or changed an event count as its antecedent, not a message that merely
  mentions it.
- **Range questions are unchanged.** `我下周有什么安排？` goes through Phase 9
  local recall, which never calls the model and still uses `date_tool`.

## Left for the separate cleanup

Per decision 3, nothing was deleted from `time_tool.py` or `date_tool.py`. These
functions now have no caller in `src/` and are kept alive only by their own tests:
`time_tool.with_stated_period`, `parse_end_time`, `stated_timezone`,
`parse_duration`, and `date_tool.states_recurrence`. `time_tool.parse_time` is
still used by the legacy V2 Phase 3 `router_service`.

## Follow-up: the current local time

The router received only the date (`datetime.now(tz).date()`), so the model could
not resolve a time relative to now, or tell that a clock time today had already
passed. A tool the model calls for the current time was considered and rejected:
the value is small and needed on every turn, so injecting it keeps one model call
per turn, cannot be skipped by the model, and stays deterministic in tests.

The UI now reads the local clock once per turn (`configured_reference_now`) and
passes both the date and `reference_time` (`HH:MM`) to the router, which forwards
it to the model. The prompt resolves phrases such as `半小时后` from it, and asks
the model to clarify when a message schedules something for a clock time today
that is already earlier than now. Python adds no rule of its own: this is
understanding, not authorization.

`gpt-5.6-luna`, reference date 2026-09-12:

| Local time | Input | Result |
| --- | --- | --- |
| 14:10 | `半小时后开个短会` | event 14:40 |
| 14:10 | `两个小时后跟客户打电话` | event 16:10 |
| 21:00 | `今天早上八点有个站会` | clarify: already earlier than now |
| 21:00 | `明天早上八点有个站会` | event 2026-09-13 08:00 |
| 16:30 | `下午三点有个评审` | clarify: already earlier than now |

Because the prompt changed, the frozen Phase 10 acceptance was re-run with
`gpt-5.6-terra` at `33fb938`:

| Run | Result |
| --- | --- |
| `runs/phase11-reference-time-20260912` | 14 of 16 |
| `runs/phase11-reference-time-20260912-b` | **16 of 16** |

In the first run the `tracker_change` turn returned `model_or_validation_failure`
after 1.86 seconds — not a timeout — so no task was created and the project
retrieval case that depends on it failed too. Replayed twelve times from that
run's exact pre-message state, the turn produced a valid task every time, and the
full rerun passed. The failing output could not be classified: the router discards
the validation error behind its generic reply, and the trace deliberately keeps no
raw model output. Recording the error type and message, without the raw output,
would make the next intermittent failure diagnosable.

## Claim boundary

One frozen acceptance run and one probe per input show the normalized contract
works on the cases recorded here. They do not measure how often the model
misreads a date, or how often a new representation will meet a rule that
predates it.
