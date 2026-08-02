# Beeja LM — Claude Code Starter Kit

**First model release:** `Beeja-3M`  
**Repository name:** `beeja-lm`  
**Tagline:** *A small language model grown from first principles.*

Beeja-3M is the first seed model in the Beeja family: a roughly three-million-parameter decoder-only Transformer implemented and trained from random initialization. The repository can later grow into `Beeja-10M`, `Beeja-30M`, and instruction-tuned variants without changing its identity.

It does **not** contain a prebuilt model implementation. It gives Claude Code the project rules, architecture targets, staged roadmap, verification gates, and reusable slash-command skills needed to build the repository carefully.

## What is included

- `CLAUDE.md` — rules Claude loads in every project session.
- `.claude/skills/beeja-lab/` — main implementation orchestrator.
- `.claude/skills/beeja-next/` — continue the next unfinished milestone.
- `.claude/skills/beeja-verify/` — verify correctness before progressing.
- `.claude/skills/beeja-lesson/` — teach one concept and connect it to the code.
- `PROJECT_STATE.md` — persistent milestone tracker.
- `LEARNING_LOG.md` — explanations, experiments, and observations.
- `MASTER_PROMPT.md` — standalone fallback prompt when skills are unavailable.
- `scripts/check_environment.py` — inspect Python, Apple Silicon, PyTorch, and MPS availability.

## Start on your Mac

```bash
unzip beeja-lm-claude-kit.zip
cd beeja-lm-claude-kit
chmod +x start.sh
./start.sh
```

Or start manually:

```bash
cd beeja-lm-claude-kit
git init
claude
```

Inside Claude Code, confirm the project files are loaded:

```text
/context
```

Then begin:

```text
/beeja-lab bootstrap
```

## Main commands

```text
/beeja-lab bootstrap
/beeja-lab bigram
/beeja-lab transformer
/beeja-lab tokenizer
/beeja-lab pretrain
/beeja-lab modernize
/beeja-lab chat
/beeja-lab evaluate
/beeja-lab mlx
/beeja-lab app
/beeja-lab polish
```

Continue from the current state:

```text
/beeja-next
```

Verify a stage:

```text
/beeja-verify transformer
```

Study a concept in the context of the repository:

```text
/beeja-lesson causal-self-attention
/beeja-lesson cross-entropy
/beeja-lesson byte-pair-encoding
/beeja-lesson rope
```

You can also invoke a skill directly from the shell as the initial prompt:

```bash
claude "/beeja-lab bootstrap"
claude "/beeja-next"
```

## Recommended working rhythm

1. Run one implementation stage.
2. Read the generated stage report and `LEARNING_LOG.md`.
3. Run `/beeja-verify <stage>`.
4. Fix every failed quality gate.
5. Commit the milestone.
6. Run `/beeja-next`.

Do not ask Claude to generate all stages in one pass. The purpose of this repository is to understand and verify every component.

## Training policy

Claude may run tiny smoke tests and short overfitting tests locally. It must not automatically launch long local or Colab training jobs. For full training, it should prepare a resumable command or notebook, explain expected resource use, and wait for an explicit instruction to run it.

## Later advanced target

The final educational model should be a 20–50M parameter decoder-only Transformer using a custom byte-level BPE tokenizer, RoPE, RMSNorm, SwiGLU, causal self-attention, instruction tuning, evaluation, checkpointing, and local inference.
