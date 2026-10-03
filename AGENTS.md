# Standing collaboration instructions

## Before changing training or strategy

The user requires lessons from prior failures to guide every optimization. Before implementing a training/strategy change:

1. Review the relevant failure evidence in `docs/experiment_history.md` (the `replay-evidence` and `plan-comparison` sections), and the current run. Distinguish measured facts, structural code limitations, and untested causal hypotheses.
2. Compare the proposed behavior with DECEM replay evidence: continuous production, conditional renewal/conversion, shared material/work routes, and cash turnover. Do not claim to know DECEM's training algorithm.
3. Check current code before reusing historical diagnoses. The current plan pipeline already has batch production, economic routes, and two-stage transitions; the old mixed pipeline's scale limits are not automatically current limits.
4. Explain which concrete missing capability changes and how collection, labels, execution, and deployment agree. Do not substitute a new version name, opponent pool, extra actions, or longer training for this explanation.
5. Preserve accepted behavior and existing checkpoints. New learners must not silently take over an entire game. Keep validation seeds out of training; label comparisons with the continuation-policy version. Do not claim incompatible checkpoints can resume unchanged.
6. Verify targeted execution/learning invariants. Small smoke tests prove connectivity only; do not report them as evidence of economic improvement. The user normally runs training experiments after receiving the command.

Historical event-plan design is consolidated in `docs/experiment_history.md` (the `event-design` section). The original proposal is not an instruction to start work without considering the user's current request; check current code for implemented capabilities.

## Competition objective

For the active event training pipeline, optimize terminal match score only: win=1, draw=0.5, loss=0 at the end of the 720-turn game. Absolute cash and cash margin are diagnostics, not auxiliary rewards or promotion vetoes. Preserve paired independent confirmation and accepted deployments; version the learning objective and recompute cached labels from actual outcomes when changing it.

## Event execution checks learned from the final audit

- Inspect actual legal candidate sets, not only event/branch counts. The first paired smoke had 11 follow-up comparisons but all were Keep/Cancel because daily workers had expired before maintenance rehired them. Use real post-maintenance observations; do not invent future workers or suppress cancellation during genuine persistent shortages.
- A route-blocked follow-up event must survive ordinary scope exhaustion until the same live batch is editable; normal dispatch must not delete or steal the reserved event.
- When event timing or execution changes, invalidate dependent comparison labels and version the contract. Preserve accepted checkpoints; never silently reinterpret accepted old event scopes.
