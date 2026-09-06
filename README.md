# CaptureFlow

CaptureFlow is a planned local, single-user AI capture inbox. It accepts pasted
plain text and routes it through a bounded plan to produce only relevant
summaries, task proposals, and long-term-memory proposals.

## Current status

The repository currently contains the preflight design only. No application,
agent behavior, evaluation result, or deployment is implemented or claimed yet.

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

The project has no Python dependencies of its own yet. Dependencies will be
declared when Phase 1 begins, rather than preemptively inheriting an
environment-specific package list.
