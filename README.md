# CaptureFlow

CaptureFlow is a local, single-user assistant under staged development. The
current implementation accepts pasted text and routes it through a bounded,
source-grounded capture workflow with policy-controlled long-term memory.

The next product iteration is intentionally narrower and chat-first: a user
will send plain-text messages, and CaptureFlow will create local SQLite-backed
calendar events or save explicit ideas and notes. This V2 direction is planned,
not implemented in the current application. It does not include external
calendar synchronization, email, meetings, vector retrieval, or general
chatbot behavior.

## Current status

Phases 1–3 of the original capture workflow are complete. Phase 2 passed 36
scripted tests and the frozen article, email, and idea live checks with retained
SQLite traces. Its evidence and claim boundary are in
[docs/phase-2-evidence.md](docs/phase-2-evidence.md). A locally retained Phase 3
acceptance run recorded 12/12; those local acceptance records are not committed
to the repository. Memory reviews support approval, edit, rejection, and
controlled conflict supersession. The current Streamlit application is still
the capture-oriented development shell; the V2 chat interface and local
calendar/idea/note actions have not been implemented yet.

Runtime source safety is separated from offline output-quality evaluation in
[docs/gate-contract.md](docs/gate-contract.md). The fixed Phase 2 article,
email, and idea checks are recorded in
[docs/phase-2-live-fixtures.md](docs/phase-2-live-fixtures.md).

The completed original sequence was:

1. Phase 1: deterministic Streamlit, typed models, and SQLite persistence.
2. Phase 2: bounded Coordinator and Organizer flow with grounded output.
3. Phase 3: cross-conversation memory and a recorded 12/12 fixed acceptance run.

The optional original Phase 4 evaluation is not a prerequisite for V2. The V2
implementation will proceed from the Phase 3 baseline in four gated increments:

1. additive SQLite migrations and new chat/event/item models;
2. deterministic persisted chat and management pages;
3. a bounded router plus local event, idea, and note tools;
4. a small recorded acceptance set and updated handoff documentation.

Existing Phase 1–3 tests remain regression requirements while V2 is developed.

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
