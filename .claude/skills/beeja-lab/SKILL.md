---
name: beeja-lab
description: Build one verified stage of the Beeja language-model learning project on Apple Silicon or Colab. Use only when explicitly asked to implement a project stage.
argument-hint: "[bootstrap|bigram|transformer|tokenizer|pretrain|modernize|chat|evaluate|mlx|app|polish]"
arguments:
  - stage
disable-model-invocation: true
effort: high
---

# Beeja Lab Stage Orchestrator

Requested stage: `$stage`

## Required context

Read these files before editing:

1. `CLAUDE.md`
2. `PROJECT_STATE.md`
3. `${CLAUDE_SKILL_DIR}/references/roadmap.md`
4. `${CLAUDE_SKILL_DIR}/references/architecture.md`
5. `${CLAUDE_SKILL_DIR}/references/testing.md`

Read `${CLAUDE_SKILL_DIR}/templates/stage-report.md` before writing the final report.

## Stage selection

- Accept only: `bootstrap`, `bigram`, `transformer`, `tokenizer`, `pretrain`, `modernize`, `chat`, `evaluate`, `mlx`, `app`, or `polish`.
- If `$stage` is missing, choose the first unfinished stage from `PROJECT_STATE.md`.
- If the requested stage depends on unfinished earlier stages, implement only the missing prerequisite that blocks it and explain why.
- Never mark a stage complete without acceptance evidence.

## Execution protocol

1. Inspect relevant files, tests, configuration, and Git status.
2. Define the milestone boundary and measurable acceptance criteria.
3. Produce a short implementation plan.
4. Implement the smallest complete vertical slice for this stage.
5. Add focused tests and meaningful assertions.
6. Run the smallest relevant test first, then the broader suite.
7. Run only short smoke training locally. Do not start full pretraining automatically.
8. Fix failures rather than weakening tests or hiding errors.
9. Update `PROJECT_STATE.md` with factual results.
10. Update `LEARNING_LOG.md` with concepts, equations, tensor shapes, and measured observations.
11. Return the stage report in the template format.

## Constraints

- Use Python and PyTorch for the main educational implementation.
- Select device in this order: MPS, CUDA when running remotely, then CPU.
- Do not assume an 8 GB Mac can train the final model efficiently.
- Do not introduce orchestration frameworks, distributed systems, Docker, APIs, databases, or web UI before their roadmap stage.
- Do not import a ready-made GPT architecture.
- External packages require a clear reason and must be added to project metadata.
- Dataset downloads must be scripted and reproducible.
- Checkpoints and training must be resumable.
- Never fabricate successful training, benchmark numbers, or test output.
