# Phase 10 live acceptance evidence

## Result

The frozen Phase 10 contextual-router fixture passed once on 2026-09-09 after
explicit API-cost approval. It used `gpt-5.6-terra` through the default
OpenAI-compatible chat-completions endpoint, with a 30-second timeout and the
`America/Vancouver` fixture timezone.

- Local redacted trace: `artifacts/phase10-live-20260909-rerun2/trace.json`
  (ignored by Git)
- Fixture: `tests/fixtures/phase10_live_acceptance.json`
- Working tree at execution: dirty; the trace records source fingerprints
  because the ambiguity safety repair was not yet committed
- Live result: 16 ordered case/assertion records passed; 7 model calls;
  18.666 seconds cumulative model time

The isolated runner creates only a local SQLite database. It does not access
an external calendar or retain credentials, prompts, raw model output, or
assistant reply text in its trace.

## Covered behavior

| Case | Observed intent/outcome | Verified bounded effect |
| --- | --- | --- |
| Calendar creation | `calendar_create` | One committed local event |
| Calendar correction | `calendar_update` | One audited update, with the requested time range |
| Task distinction | `tracker_change` | One committed Project record, not an event |
| Meeting question | `local_answer` / grounded local context | No write; stored local source card |
| Replay | `reloaded` | Zero model calls and no duplicate event |
| Cross-conversation meeting lookup | `local_answer` / grounded local context | No write; stored local source card |
| Cross-conversation Project lookup | `project_answer` | No write; stored Project source card |
| Ambiguous calendar correction | `calendar_update` proposal routed to `clarification_created` | No event update; pending clarification saved |

## Failure retained and repair

The first run, `artifacts/phase10-live-20260909-rerun1/trace.json`, made seven
model calls and failed exactly one assertion: `ambiguous_correction_effect`.
The model proposed `calendar_update` for a correction containing a bare
weekday. The date tool rejected that expression and the dispatcher returned
`write_or_dispatch_failure`, rather than creating a clarification.

The repair is deterministic: an update proposal with a non-resolvable date
expression now creates a pending date clarification without changing Calendar
state. The corresponding scripted test and the unchanged frozen live fixture
then passed on the rerun above. The failed trace remains in the ignored local
artifacts directory; it was not replaced or deleted.

## Deterministic evidence

The non-UI suite ran as 174 tests and passed. The 20 Streamlit `AppTest` cases
also passed when run in four isolated groups; this grouping is necessary on
this Windows host because one large `AppTest` process does not reliably return
an exit marker even after its test work has finished. The isolated groups are
an execution-environment workaround, not a change to test assertions.

## Executed-input fingerprints

| Path | SHA-256 |
| --- | --- |
| `src/captureflow/conversation_router_service.py` | `60c58c1d3708e5f6439bdbc08be4c9b7cc4b769e6ec18ffaabc58b0883e32c78` |
| `src/captureflow/llm.py` | `db46be80ca0a393695b8c483d3252049b1ec28aeda038b9ddc78bf3a84b396ca` |
| `src/captureflow/ui.py` | `fe48e92e3a27f463fc32c57f5b2a59f812aebe6b3723b2510d7342b27ef6ab1e` |
| `scripts/run_phase10_live_acceptance.py` | `380d2938d9553b45c71fa91223989f58371b3000ff7ff81b2e39281a6c3a8187` |
| `tests/fixtures/phase10_live_acceptance.json` | `da6c6f4394d068e5c6a11f3760032bb1b79093f225a35af293621f3422594fb4` |

## Claim boundary

This is one fixed-sequence demonstration, not a routing reliability study or
a retrieval benchmark. The deterministic tests establish validation,
persistence, idempotency, and local safety boundaries. The live run only shows
that this exact model, prompt, fixture, and code snapshot completed the stated
sequence once; it does not establish performance on unseen language or permit
external-calendar claims.
