# Moving the router to Qwen

The OpenAI account behind `gpt-5.6-terra` ran out of credits after the
[redundancy cleanup](redundancy-cleanup.md), so the frozen Phase 10 acceptance
could not confirm the cleanup. Daily use moved to Qwen through the DashScope
international endpoint. This records what that exposed, what was fixed, and the
evidence so far.

## Decisions

- Qwen becomes the default and acceptance model. The fixture's protocol model is
  now `qwen3.8-flash`; its eight cases and their checks are unchanged.
- Diagnose before fixing: a failed turn must say why.
- A `project_answer` that cites Calendar events or ideas/notes is answered by the
  deterministic local lookup, so event times come from SQLite.
- Whether `cross_conversation_local_retrieval` should check the model's intent or
  the answer's effect is deferred. The cleanup made the two differ, and a Qwen
  `project_answer` there may point at a flaw in the case, not in the router.

## First Qwen runs, before any fix

Two `qwen3.6-plus` runs, with the model overridden, passed 14 and 12 of 16
records: `runs/cleanup-phase10-live-20260912-qwen36-codex` at `e853a6c`, and
`-b` at `d96f6d5`. A failed turn reported only `model_or_validation_failure`,
because the router discarded the error, so the cause could not be seen.

## Fixes

| Commit | Change |
| --- | --- |
| `27ee62c` | A failed turn records its error type, field path, and rule in Diagnostics and the acceptance trace, never model output |
| `7092ab8` | `recurring: null` means not repeating; the prompt asks for null on unstated fields, but the schema rejected it |
| `e3bff8d` | A `project_answer` citing Calendar or idea/note sources goes to the local lookup |
| `ef96b25` | Launcher and fixture default to Qwen: `QWEN_API_KEY`, DashScope international endpoint, 120-second timeout |
| `3050f9e` | Payload-rule errors name the offending fields, by name only |
| `2b9d914` | `source_ids` on a write, tracker change, or clarification is dropped instead of rejecting the decision |
| `84c49d6` | A `calendar_update` with no new value becomes a date clarification instead of a schema failure |
| `750bde5` | `qwen3.8-flash` replaces `qwen3.6-plus` as the default and protocol model |

Four of these share one shape, already noted in Phase 11: a structural rule that
was stricter than the prompt's contract ran before the policy meant to handle the
case. `gpt-5.6-terra` happened to satisfy those rules; other models do not.

## Choosing a cheaper model

Each candidate ran the same three real router turns in a project conversation: a
tracker task, a calendar creation, and a meeting question. Prices are the listed
Singapore rates per million input and output tokens at the time. Each routing call
sends about 4,000 input tokens.

| Model | Thinking | Right intent | Latency per turn | Failure seen | Price in / out |
| --- | --- | --- | --- | --- | --- |
| `qwen3.6-plus` | default | 1 of 3 | 11–54 s | cited a source outside the candidates | $0.50 / $3.00 |
| `qwen3.8-flash` | default | **3 of 3** | 8–23 s | — | $0.15 / $0.47 |
| `qwen3.7-flash` | default | 0 of 3 | 13–21 s | duplicate evidence; missing content | tiered, from $0.03 / $0.13 |
| `qwen3.5-flash` | default | 2 of 3 | 4–44 s | tracker payload rule | $0.10 / $0.40 |
| `qwen-turbo` | default | 0 of 3 | 2–3 s | chose `reply_only`; payload rules | $0.05 / $0.20 |
| `deepseek-v4-flash` | default | 0 of 3 | 5–14 s | `project_changes: null` | $0.20 / $0.40 |

With thinking disabled (`enable_thinking: false`), every Qwen model answered in
1–6 seconds and used far fewer output tokens, but none chose `tracker_change` for
the tracker turn. Thinking is therefore left at the provider default. The screen
ran before `2b9d914` and `84c49d6`, so a few of these failures would now pass.

## Acceptance with `qwen3.8-flash` at `750bde5`

| Run | Result | Model calls |
| --- | --- | --- |
| `runs/qwen38flash-protocol-20260912-a` | 11 of 16 | 7 |
| `runs/qwen38flash-protocol-20260912-b` | **16 of 16** | 7 |

Run `a` failed at its first turn: the model gave the meeting both an end time and
a duration, which the schema rejects, so no event was created. The four cases that
depend on that event failed with it: the update clarified, the meeting question
cited a source that did not exist, the replay found no event, and the
cross-conversation lookup had nothing to cite. Run `b` passed every record, with
`ambiguous_correction` reaching `clarification_created` through the target rule.

Overridden `qwen3.8-flash` runs at intermediate commits scored:

| Commit | Result | Note |
| --- | --- | --- |
| `8c07e0e` | 12 of 16 | Before the failure reasons named their fields |
| `3050f9e` | 14 of 16 | The update failed on `source_ids` |
| `2b9d914` | 15 of 16 | The ambiguous correction failed for lack of a new value |

## Open issues

- **Run-to-run variance is high.** One structural slip at the first turn fails
  five records. Two runs cannot measure a pass rate.
- **End time plus duration.** A model that states both, consistently, is rejected.
  Accepting both when they agree, and clarifying when they conflict, would follow
  the same pattern as the fixes above. Not done yet.
- **Latency.** Routing calls took 8–100 seconds. The OpenAI SDK retries a timed-out
  request twice by default, so one turn reached 169 seconds at a 120-second
  timeout, and a `qwen3.8-flash` turn at `8c07e0e` reached 362 seconds before timing out. The Chat UI
  waits that long.
- **Deferred decision.** Whether `cross_conversation_local_retrieval` should check
  intent or effect.

## Claim boundary

These are a handful of runs on one fixed sequence. They show that each fixed
failure no longer occurs on the turns observed, and that `qwen3.8-flash` can pass
the full acceptance. They do not show how often it will.
