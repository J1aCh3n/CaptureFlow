# Phase 1 gate evidence

**Status:** passed on 2026-09-06.

## Scope assessed

Phase 1 provides a deterministic local Streamlit shell, typed `Conversation`
and `Capture` models, and SQLite persistence. It deliberately contains no
model provider, Agent, plan, task extraction, memory retrieval, or generated
content.

## Verification

Environment: `llm-course` (Python 3.11.15, Streamlit 1.59.2, Pydantic 2.13.4).

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
python -m compileall -q app.py src tests
```

Result: 8 tests passed.

The suite verifies:

- blank and over-limit manual captures are rejected;
- a saved capture can be read after a new `CaptureStore` instance opens the
  same database;
- repeated saves with the same ID do not duplicate a row;
- repeated deletes are safe;
- the Streamlit shell renders with its no-model boundary; and
- a hand-authored capture can be saved and deleted through the UI.

`streamlit run app.py --server.headless true --server.port 8510` also started
successfully and its local health endpoint returned `ok` during verification.

## Remaining boundary

This evidence passes only the Phase 1 gate. It is not evidence of Coordinator,
Organizer, tool, grounding, task, memory, live-model, or evaluation quality.
