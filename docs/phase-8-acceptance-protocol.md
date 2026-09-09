# Phase 8 tracker acceptance protocol

The frozen sequence is
[`tests/fixtures/phase8_tracker_acceptance.json`](../tests/fixtures/phase8_tracker_acceptance.json).
It fixes the nine required acceptance cases, an initial local database seed,
and the timezone before any live run.

The scripted test exercises this sequence without an API call. It proves the
application flow, SQLite state transitions, project isolation, source cards,
and restart behavior—not a model's ability to select the intended action.

Before a live run, record all of the following beside the resulting local
database and transcript:

- date/time, configured model ID, endpoint family, and timeout;
- checked-out commit and this fixture's contents;
- timezone and database seed;
- each user input, assistant reply, proposed/committed/reviewed operation,
  source record IDs, and any failure;
- the full ordered sequence, including failures. Do not replace failed cases
  with easier prompts.

Running that live acceptance costs model tokens. It requires explicit user
approval before execution. A single successful run is demonstration evidence,
not a reliability or retrieval benchmark.

After approval, use `scripts/run_phase8_live_acceptance.py` with a new output
directory. It creates an isolated SQLite database and writes a `trace.json`
containing the configured model metadata, fixture, ordered cases, outcomes, and
failures. The output directory is intentionally ignored by Git so credentials
and local run material are not committed accidentally.

The runner also records whether the working tree was dirty and SHA-256
fingerprints of the model adapter, tracker service, retrieval service, and
frozen fixture. It evaluates the sequence itself: a process exit code of zero
means every assertion passed and no case recorded an error. It still does not
make one run a reliability claim.
