# Phase 11 Design: Model-Normalized Dates and Times

## Status

Implemented on 2026-09-12; see [phase-11-evidence.md](phase-11-evidence.md). The open questions
are answered in the Decisions section at the end.

## Summary

The contextual router asks the model to copy raw date and time words from the
message (`date_expression: "明天"`, `time_expression: "下午四点"`), and Python
turns those words into timestamps. That makes Python responsible for understanding
every way a person states a time, in two languages.

This phase moves that understanding to the model. The model returns normalized
fields — `date: "2026-09-13"`, `start_time: "15:00"`, `end_time: "16:00"` — and
Python checks their structure, attaches the configured timezone, and keeps every
rule that decides whether a write is allowed.

The principle this phase adopts:

> **Understanding belongs to the model. Authorization stays in Python.**

## Why

### Where the current design came from

It was inherited, not chosen against this alternative.

- **V1** (`PREFLIGHT.md`) was a multi-agent capture pipeline for pasted articles
  and emails. One of its stated goals was typed tool use —
  "Three model-selectable, typed tools: `resolve_relative_date`, …" — and its
  definition of done required that "relative dates are resolved through a typed
  tool or reported as ambiguous." For English email deadlines with a small fixed
  vocabulary, that was reasonable.
- **V2** (`PREFLIGHT_V2.md` §8) turned the product into free-form bilingual chat
  and specified only "Reuse the existing deterministic date resolver."
- **Phase 10.3** later justified keeping it: asking the model to turn `半个小时`
  into `19:30` "would move arithmetic and normalization into the model and weaken
  the deterministic evidence boundary."

### What it costs now

The concerns behind the tool were real: a model can invent a date nobody stated,
and it can get date arithmetic wrong. Neither concern requires Python to
understand language, yet that is where the work went. Phases 10.4–10.8 added
Python rules for `中午`, bare `十二点`, `9月20日`, English month names,
`一个半小时`, `an hour`, `下午四点到五点`, a period the model split away from its
hour, named timezones, and repeating schedules.

| File | Lines at `504f39f` | Lines now |
| --- | --- | --- |
| `time_tool.py` | 83 | 240 |
| `date_tool.py` | 241 | 316 |
| `conversation_router_service.py` | 787 | 1099 |

The frozen Phase 10 rerun shows the treadmill directly. Across five runs,
`gpt-5.6-terra` split the same inputs three different ways — `下午四点` / `五点`,
`下午两点` / `三点`, and `两点` / `三点` with `下午` in neither field — and each new
split needed a new Python rule. The rules were chasing how the model copies text,
which is not a problem Python can finish solving.

## The new contract

### `ContextualAction`

Replace the four expression fields with normalized fields:

| Remove | Add | Format |
| --- | --- | --- |
| `date_expression` | `date` | `YYYY-MM-DD` or `null` |
| `time_expression` | `start_time` | `HH:MM`, 24-hour, or `null` |
| `end_time_expression` | `end_time` | `HH:MM`, 24-hour, or `null` |
| `duration_expression` | `duration_minutes` | integer or `null` |
| — | `timezone` | IANA name the message states, or `null` |
| — | `recurring` | `true` when the message states a repeating schedule |

Schema rules, enforced by the Pydantic model:

- `null` means the message does not state it. The model must never fill a field
  the message does not support.
- `end_time` and `duration_minutes` are mutually exclusive, and either requires
  `start_time`.
- `start_time` requires `date`.
- `duration_minutes` is between 1 and 720.
- All existing rules about `content`, `description`, `target_event_id`, and
  `evidence` are unchanged.

For `明天下午三点到四点` with reference date 2026-09-12:

```json
{
  "action": "create_event",
  "content": "…",
  "date": "2026-09-13",
  "start_time": "15:00",
  "end_time": "16:00",
  "duration_minutes": null,
  "timezone": null,
  "recurring": false,
  "evidence": [{"message_id": "…", "source_quote": "…"}]
}
```

### Why date and time are separate, and the timezone is Python's

The model never writes a UTC offset. Python combines `date` and `start_time` with
the configured zone through `zoneinfo`. Offsets change within a year — in
`America/Vancouver`, 2026-10-31 is `-07:00` and 2026-11-02 is `-08:00` — and that is
arithmetic `zoneinfo` gets right every time, while a model writing an ISO offset
for a date past 1 November can easily get it wrong.

### `ProjectChange`

| Remove | Add |
| --- | --- |
| `due_date_expression` | `due_date` (`YYYY-MM-DD` or `null`) |
| `due_time_expression` | `due_time` (`HH:MM` or `null`) |

`due_at` stays Python-only, and the router still refuses a model-supplied
`due_at`. `set_deadline` keeps its shape with the new fields.

## What the model receives

In addition to today's payload, a small calendar table so that weekday phrases
are looked up rather than computed:

```json
"calendar": [
  {"date": "2026-09-05", "weekday": "Saturday", "weekday_zh": "周六"},
  "…",
  {"date": "2026-10-09", "weekday": "Friday", "weekday_zh": "周五"}
]
```

The table spans the reference date minus 7 days to plus 27 days: last week through
roughly four weeks ahead, which covers `上周五`, `下周三`, and `下下周三`. A date
outside it, such as `12月1日`, is still an ordinary month and day for the model.

The prompt's copy-the-expression instructions are replaced with: use the calendar
table for weekday phrases; return `null` for anything the message does not state;
report a stated timezone and a repeating schedule in their fields rather than
converting or picking one occurrence.

## What Python does

### Structure

- Parse `date` and times; an unparseable value fails validation.
- Attach the configured timezone.
- An `end_time` at or before `start_time` ends the next day, as today.
- A duration adds minutes to the start, as today.
- A `date` more than one year before or two years after the reference date is
  implausible and asks for the date.

### Policy on the new fields

| Field | Python response |
| --- | --- |
| `timezone` set and not the configured zone | `timezone` clarification, no write |
| `recurring: true` | `recurrence` clarification, no write |
| `create_event` with `date: null` | `date` clarification, no write |

These keep the existing `PendingClarification.missing_field` values, so no
migration is needed.

### Authorization — unchanged

Every rule that decides whether a write may happen stays exactly as it is:
evidence quotes, candidate membership, target grounding in the conversation,
the credential guard, description quoting, one action per turn, deferred
follow-ups, no-op updates, pending-question replacement and resolution, and the
refusal of a model-supplied `due_at`.

The credential guard must stay deterministic for a different reason than the
others: it runs before any model call, because its whole job is that the text
never leaves the machine.

### Echo-back

Because Python can no longer fail closed on a misread time, every write reply
states the resolved absolute schedule, so the user sees a misreading immediately
and can undo it. Creation and deadline replies already do; the update reply
changes from `Updated local calendar event: X.` to include the new schedule.

## The trade-off being accepted

**Ambiguity no longer fails closed in Python.** Given `提醒我明天十点吃药`, Python
cannot tell whether the model resolved `十点` from context or guessed. The prompt
tells the model to return `null` and clarify when the message itself does not
determine the part of the day, but Python cannot verify that. The mitigations are
echo-back, Undo, and the live probes below.

**A weekday miscount becomes a wrong write, not a clarification.** The calendar
table is the mitigation.

**This reverses a recorded boundary.** Phase 10.3's exit gate "resolved by Python,
not model arithmetic" and several Phase 10.7 fail-closed rows no longer hold. That
reversal is intentional and should be recorded when this phase is accepted.

Some Phase 10.7 items that were unsupported on purpose become supported as a
consequence: `下下周三`, `月底`, `九月二十日`, and relative shifts such as
`提前一小时` on an update, since the model can see the event's current time.

## What is removed and what stays

### Removed from the contextual router path

- `time_tool`: Chinese period parsing, `parse_end_time`, `with_stated_period`,
  `parse_duration`, `stated_timezone`.
- `date_tool.states_recurrence`.
- In `conversation_router_service.py`: `_with_stated_periods`,
  `_states_other_timezone`, `_resolved_date`, the expression branches of
  `_calendar_clarification_field`, and expression resolution in
  `_resolved_deadline`.

### Stays

- `date_tool.resolve_relative_date`, because Phase 9 local recall
  (`local_context_service`) answers `我明天有什么安排？` without any model call, and
  the V1 `workflow.py` pipeline still uses it.
- The legacy V2 Phase 3 `router_service.BoundedChatService` still imports
  `time_tool.parse_time`. The UI does not route through it — slash commands use
  `DeterministicChatService` and ordinary chat uses `ConversationRouterService` —
  so `time_tool.py` can only be deleted once that legacy router is retired. See
  the open questions.

## Compatibility

- **No schema migration.**
- **Stored tracker proposals.** `ProjectChange` is re-validated from stored JSON in
  `save_project_change_operation`, `commit_prepared_project_change`, and
  `undo_project_change_operation`, and it forbids unknown keys. The local database
  already holds 2 committed operations whose stored change includes
  `due_date_expression`. Removing the field outright would break Undo on those
  rows. `ProjectChange` gets a `mode="before"` validator that drops the two legacy
  keys; their resolved value already lives in `due_at`. A test loads such a row.
- **Pending clarifications** store the old action shape, but that JSON is only
  passed to the model as context and is never re-validated. The local database has
  1 pending row.
- The local database is at schema version 13; migrations 14–16 from Phase 10.7
  apply on the next start, as before.

## Tests

- Scripted decisions in `test_conversation_router_service.py` (83 references to
  expression fields), `test_models.py` (19), `test_llm.py` (8), and `test_ui.py` (4)
  are rewritten to the new fields. Their behavioral assertions — what is written,
  what clarifies, what is refused — are the contract and stay.
- Parser tests in `test_time_tool.py` that exist only for the removed router path
  are deleted. `test_date_tool.py` stays for local recall.
- `test_router_service.py` (legacy, 6 references) is untouched.
- No frozen fixture contains an expression field, so no frozen evidence changes.
- New tests: DST offset on 2026-11-02, an end time before the start rolling to the
  next day, malformed `date` and `HH:MM` values, a non-local `timezone`,
  `recurring: true`, an implausible date, the calendar table's range and weekdays,
  the update echo-back, and a legacy stored `ProjectChange` still undoing.

## Evidence plan

1. The full deterministic suite passes.
2. Frozen Phase 10 live acceptance with `gpt-5.6-terra` stays 16 of 16. Phase 8 is
   untouched by this change and is not re-run unless tracker services change.
3. A `gpt-5.6-luna` probe re-runs the Phase 10.7 boundary scenarios — `9月20日`,
   `一个半小时`, `for an hour`, `下下周三`, `月底`, `纽约时间`, `每周一`,
   `提醒我明天十点吃药`, and the two range splits that failed in Phase 10.8 — and
   records what each does now, including items that are newly supported.

## Implementation order

One commit per step:

1. Schema fields, rules, and the legacy-key validator, with model tests.
2. Calendar table and prompt rewrite, with prompt tests.
3. Router: normalized fields replace expression resolution; timezone and
   recurrence policy; update echo-back; router tests.
4. Deadlines move to `due_date` and `due_time`.
5. Remove the now-unused router helpers; full suite.
6. Live acceptance, luna probes, and the evidence record.

## Decisions

The four open questions were answered by the user on 2026-09-12.

1. **Ambiguous hour.** The model resolves a bare hour when the message's own
   context determines it (`晚饭后十点` is 22:00) and clarifies only when it cannot.
2. **Deterministic backstops.** Not kept. Timezone and recurrence are reported by
   the model in their fields; Python applies policy to those fields only. The
   credential guard is unaffected because it must run before any model call.
3. **Legacy router.** Ignored in this phase. `router_service.BoundedChatService`,
   `time_tool.py`, and their tests are left in place and cleaned up separately as
   redundant files.
4. **Calendar table range.** Reference date minus 7 days to plus 27 days.
