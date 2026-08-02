---
name: beeja-verify
description: Verify a Beeja language-model stage using focused correctness, leakage, checkpoint, and smoke tests.
argument-hint: "[stage-or-scope]"
arguments:
  - scope
disable-model-invocation: true
effort: high
---

Verify scope: `$scope`

1. Read `CLAUDE.md`, `PROJECT_STATE.md`, and `.claude/skills/beeja-lab/references/testing.md`.
2. Inspect the implementation before trusting existing tests.
3. Map the requested scope to explicit invariants and failure modes.
4. Run focused tests first, then the relevant broader suite.
5. Add missing tests only when they validate real behavior rather than implementation details.
6. For Transformer code, explicitly check causal leakage and tensor shapes.
7. For training code, run only a short smoke or tiny-overfit test.
8. For checkpoints, verify model, optimizer, scheduler, scaler, step, config, and RNG state when applicable.
9. Never mark a test as passed without executing it.
10. Update `PROJECT_STATE.md` only with actual evidence.

Return:

- Verdict: pass, partial, or fail.
- Invariants checked.
- Commands and concise actual results.
- Defects found and fixes made.
- Remaining unverified assumptions.
- One recommended next command.
