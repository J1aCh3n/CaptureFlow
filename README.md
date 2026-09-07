# CaptureFlow

CaptureFlow is a local, single-user AI capture inbox under staged development.
It accepts pasted plain text and routes it through a bounded plan. Phase 2
produces source-grounded summaries and task proposals; long-term-memory
proposals remain Phase 3 work.

## Current status

Phases 1 and 2 are complete. Phase 2 passed 36 scripted tests and the frozen
article, email, and idea live checks with retained SQLite traces. The evidence
and claim boundary are in
[docs/phase-2-evidence.md](docs/phase-2-evidence.md). Phase 3 implementation
is in progress; its tested memory-policy foundation and remaining gate work are
in [docs/phase-3-evidence.md](docs/phase-3-evidence.md). In particular, a
visible confirmation note cannot yet receive a reply, confirm, edit, or
task-update action.

Runtime source safety is separated from offline output-quality evaluation in
[docs/gate-contract.md](docs/gate-contract.md). The fixed Phase 2 article,
email, and idea checks are recorded in
[docs/phase-2-live-fixtures.md](docs/phase-2-live-fixtures.md).

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

## Phase 2 model configuration

Analysis is opt-in: pressing **Analyze** sends the capture to a configured
OpenAI-compatible chat-completions endpoint. Set these environment variables
in the shell that starts Streamlit:

```powershell
$env:CAPTUREFLOW_API_KEY = "..."
$env:CAPTUREFLOW_MODEL = "your-model-id"
# Optional for a compatible provider endpoint:
$env:CAPTUREFLOW_BASE_URL = "https://provider.example/v1"
# Optional; defaults to America/Vancouver:
$env:CAPTUREFLOW_TIMEZONE = "America/Vancouver"
streamlit run app.py
```

The configured endpoint must support chat-completions JSON-object responses.
Without the required key and model variables, Analyze saves and displays a
typed configuration failure; it never fabricates a result.
