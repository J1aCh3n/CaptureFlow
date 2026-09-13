# Task: re-run the frozen Phase 10 live acceptance after the redundancy cleanup

**Status:** open for the frozen protocol model. Two explicitly overridden
`qwen3.6-plus` comparisons completed 14 of 16 records at `e853a6c` and 12 of 16
at `d96f6d5`; the rerun reproduced the missing Tracker record and added a
meeting-query intent mismatch plus an ambiguous-correction validation failure.
Delete nothing, and change no product code, fixture, or prompt as part of this
task.

## Why

The [redundancy cleanup](redundancy-cleanup.md) restructured
`ConversationRouterService.submit` and replaced the local-recall date lookup. The
deterministic suite (159 tests) passed after every commit, but the frozen live
acceptance could not run at `3dd73b3`: every model call returned
`429 insufficient_quota` because the API account had no credits. This task
supplies the missing live evidence.

The router prompt and the decision schema did not change, so the expected result
is the same **16 of 16** recorded in
[phase-10.8-frozen-acceptance-rerun.md](phase-10.8-frozen-acceptance-rerun.md) and
[phase-11-evidence.md](phase-11-evidence.md).

## Environment

- Worktree: `C:\Users\frank\OneDrive\Documents\Uni\gituhb_project\captureflow-contextual-router`
- Branch: `codex/contextual-router-phase10`. Do not merge into or push to `main`.
- Python: `C:\Users\frank\anaconda3\envs\llm-course\python.exe`. Do not create a
  venv or `uv.lock`; `pytest` is not installed, and the project uses `unittest`.
- The environment has an editable install pointing at the main checkout, so set
  `PYTHONPATH` to this worktree's `src` in every shell. Check that
  `captureflow.__file__` resolves inside this worktree.
- API use is approved for this run. The key comes from `REPOMEDIC_OPENAI_API_KEY`.
- This task still requires the fixture model, so `CAPTUREFLOW_MODEL` must be
  `gpt-5.6-terra`. The runner accepts another configured model only with the
  explicit `--allow-model-override` flag and records that run as a cross-model
  comparison; such a run does not complete this frozen acceptance task.

```powershell
$wt = "C:\Users\frank\OneDrive\Documents\Uni\gituhb_project\captureflow-contextual-router"
Set-Location $wt
$py = "C:\Users\frank\anaconda3\envs\llm-course\python.exe"
$env:PYTHONPATH = "$wt\src"
$env:CAPTUREFLOW_API_KEY = $env:REPOMEDIC_OPENAI_API_KEY
$env:CAPTUREFLOW_MODEL = "gpt-5.6-terra"
& $py -c "import captureflow; print(captureflow.__file__)"
```

## Steps

1. **Start clean.** `git status --short` must be empty, because the trace records
   `working_tree_dirty`. Note the commit with `git rev-parse --short HEAD`.

2. **Deterministic gate.** It must report `OK` with 159 tests before any API call:

   ```powershell
   & $py -m unittest discover -s tests
   ```

3. **One live call first.** Confirm the account has credits before spending a
   full run:

   ```powershell
   & $py -c "from captureflow.llm import ModelSettings, OpenAICompatibleModel as M; m={'message_id':'m1','role':'user','content':'明天下午三点开评审会'}; print(M(ModelSettings.from_environment()).propose_conversation_decision(current_message=m, recent_messages=[m], candidate_events=[], candidate_saved_items=[], candidate_project_records=[], reference_date='2026-09-12', timezone='America/Vancouver').intent)"
   ```

   If it raises `ModelError` with `429` or `insufficient_quota`, **stop**. Report it
   as a billing blocker and change nothing.

4. **Run the frozen acceptance.** The output directory must not already exist; `runs/`
   is ignored by Git.

   ```powershell
   & $py scripts/run_phase10_live_acceptance.py --output-dir runs/cleanup-phase10-live-20260912-codex
   ```

   The exit code is `0` only when every record passed.

5. **Read the result** from `runs/.../trace.json`:
   - Top level: `passed`, `commit`, `working_tree_dirty` (must be `false`).
   - `cases` alternates a submit record (`name`, `outcome`, `intent`,
     `model_calls`, `elapsed_seconds`) with its `*_effect` check.
   - A submit record's `status` only means no exception was raised. **Pass or fail
     is decided by the `*_effect` records.**

   Expected submit outcomes:

   | Case | Expected `outcome` / `intent` |
   | --- | --- |
   | `calendar_create` | `calendar_create`, 16:00–17:00 event |
   | `calendar_update` | `calendar_update`, event moved to 14:00–15:00 |
   | `tracker_change` | `tracker_change`, one committed `create_record` |
   | `meeting_question` | intent `local_answer`, local source cards saved |
   | `replay_idempotency` | `reloaded`, `model_calls` 0 |
   | `cross_conversation_local_retrieval` | intent `local_answer` |
   | `cross_conversation_project_retrieval` | `project_answer` with project sources |
   | `ambiguous_correction` | intent `calendar_update`, outcome `clarification_created` (reason `target`) |

## If a record fails

Do not edit the fixture, the prompt, or the expectations to make it pass.

- **Every model turn fails in about 1–2 seconds** with `model_or_validation_failure`.
  This is the credits signature. Confirm with step 3 and stop.
- **Exactly one turn fails.** A single non-reproducible `model_or_validation_failure`
  has happened before (Phase 11 evidence: 14 of 16, then 16 of 16). Re-run the full
  acceptance once into a new output directory and report both runs.
- **The same record fails in two runs.** Treat it as a possible regression from the
  cleanup. Reproduce that one turn from the failing run's database: copy
  `runs/.../captureflow.sqlite3`, delete the messages created at or after the failing
  message, and submit the same content with the same message ID through
  `ConversationRouterService`, using reference date `2026-09-09` and timezone
  `America/Vancouver`. Log the validation error type and message. Then compare with
  the pre-cleanup router at `8626655` (for example, a temporary `git worktree add`)
  on the same database copy to establish whether the cleanup changed the outcome.
  Report the findings and do **not** fix product code in this task.

## Record the result

1. In [redundancy-cleanup.md](redundancy-cleanup.md), replace the last paragraph of
   **Evidence boundary** with the run(s):
   - output directory
   - commit
   - `N of 16`
   - total model calls
   - a one-line explanation of any failure

   Keep the existing claim-boundary tone: one run is demonstration evidence, not a
   reliability measure.

2. **Do not commit** raw model output, prompts, credentials, the trace, or the
   database. Only the summary goes into the document.

3. Set this task's **Status** to `done` with a one-line outcome, then commit both
   documents on the branch:

   ```text
   docs: record the live acceptance after the redundancy cleanup

   Co-Authored-By: <your attribution>
   ```

4. Report back:
   - the result table
   - total model calls
   - whether a re-run was needed
   - the commit hash
