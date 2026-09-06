# CaptureFlow

CaptureFlow is a planned local, single-user AI capture inbox. It accepts pasted
plain text and routes it through a bounded plan to produce only relevant
summaries, task proposals, and long-term-memory proposals.

## Current status

Phase 1 is complete: the repository contains a deterministic local capture
slice with no model, Agent, task extraction, or memory behavior. The recorded
gate evidence is in [docs/phase-1-evidence.md](docs/phase-1-evidence.md).
Phase 2 and later are not implemented.

The implementation sequence is intentionally gated:

1. Phase 1: deterministic Streamlit, typed models, and SQLite persistence.
2. Phase 2: bounded Coordinator and Organizer flow with grounded output.
3. Phase 3: cross-conversation memory and the fixed acceptance examples.
4. Optional Phase 4: portfolio hardening and development evaluation.

See [PREFLIGHT.md](PREFLIGHT.md) for the frozen scope, safety boundaries, and
acceptance gates.

## Development environment

Use the existing Conda environment:

```powershell
conda activate llm-course
```

Install the project's declared dependencies into that environment, then run the
local shell:

```powershell
python -m pip install -e .
streamlit run app.py
```

The application stores local data in `data/captureflow.sqlite3`, which is not
committed. To run the deterministic unit tests:

```powershell
python -m unittest discover -s tests -v
```
