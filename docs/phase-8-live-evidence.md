# Phase 8 live acceptance evidence

## Result

The frozen Phase 8 tracker acceptance sequence passed once on 2026-09-08 after
explicit API-cost approval. It used `gpt-5.6-terra` through the configured
OpenAI-compatible chat-completions endpoint, with a 30-second timeout and the
`America/Vancouver` fixture timezone.

- Local trace: `runs/phase8-live-20260908-223510/trace.json` (ignored by Git)
- Git revision: `101b0fa80a03c7b5bb5a3a7697deb26c5aed9c68`
- Working tree: dirty at execution; the trace includes source fingerprints to
  identify the executed inputs
- Database seed: `captureflow-phase8-self-tracker`
- Runner result: 12 assertions passed; zero runner errors

The run used an isolated SQLite database under the output directory. Neither
credentials nor raw model output are committed here.

## Covered behavior

The ordered sequence created and completed a task through a follow-up, saved a
decision and explicit rationale, placed its replacement in review and then
approved the supersession, and resolved a blocker. It then verified that a
casual turn made no record, a new conversation returned source-backed current
status and history, similarly worded state from another project stayed
isolated, and a prepared operation resumed after reopening the database without
another model call.

## Executed-input fingerprints

| Path | SHA-256 |
| --- | --- |
| `src/captureflow/llm.py` | `f80854ec7e6c037cc6a12655411407f45aaf135bac262c8ecb90c41d1b121a12` |
| `src/captureflow/project_service.py` | `68c2367c119170e44b265c3c8cb12a320ca0c628ff553d48be1fbca36789bc5a` |
| `src/captureflow/retrieval_service.py` | `ad818b779ddd1d298fb9cf85a7cd641cd6440963ee0c08d05625cc358ace63a8` |
| `tests/fixtures/phase8_tracker_acceptance.json` | `fb90347c282ee08c94ce5f0bc7e71868d5536e6b79ebf25474da24d6ae2f509d` |

## Claim boundary

This is a single fixed-sequence demonstration, not a reliability study,
retrieval benchmark, or evidence that the model will consistently select the
right tracker action on unseen conversation. The deterministic/scripted suite
remains the evidence for application invariants; future live runs must retain
all failures rather than replace difficult inputs.
