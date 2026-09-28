# Standing collaboration instructions

## Before changing training or strategy

The user requires lessons from prior failures to guide every optimization. Before implementing a training/strategy change:

1. Review the relevant failure evidence in `docs/competition_replay_review.md`, `docs/plan_compare_trial_review.md`, and the current run. Distinguish measured facts, structural code limitations, and untested causal hypotheses.
2. Compare the proposed behavior with DECEM replay evidence: continuous production, conditional renewal/conversion, shared material/work routes, and cash turnover. Do not claim to know DECEM's training algorithm.
3. Check current code before reusing historical diagnoses. The current plan pipeline already has batch production, economic routes, and two-stage transitions; the old mixed pipeline's scale limits are not automatically current limits.
4. Explain which concrete missing capability changes and how collection, labels, execution, and deployment agree. Do not substitute a new version name, opponent pool, extra actions, or longer training for this explanation.
5. Preserve accepted behavior and existing checkpoints. New learners must not silently take over an entire game. Keep validation seeds out of training; label comparisons with the continuation-policy version. Do not claim incompatible checkpoints can resume unchanged.
6. Verify targeted execution/learning invariants. Small smoke tests prove connectivity only; do not report them as evidence of economic improvement. The user normally runs training experiments after receiving the command.

Current detailed design discussion: `docs/event_plan_next_steps.md`. It is a proposal, not an implementation or an instruction to start work without considering the user's current request.
