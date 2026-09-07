# CaptureFlow

CaptureFlow is a planned local, single-user AI capture inbox. It accepts pasted
plain text and routes it through a bounded plan to produce only relevant
summaries, task proposals, and long-term-memory proposals.

## Current status

Phase 1 is complete. Phase 2's bounded planning and grounding implementation is
present, but its gate remains open pending three saved live runs. Evidence and
remaining requirements are in [docs/phase-2-evidence.md](docs/phase-2-evidence.md).
Phase 3 and later are not implemented.

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
