# Roadmap and Stage Acceptance Criteria

## 1. Bootstrap

Deliver:

- `pyproject.toml` with a minimal dependency set.
- `src/`, `tests/`, `configs/`, `scripts/`, `notebooks/`, `reports/`, and `checkpoints/` layout.
- Device-selection utility.
- Deterministic seeding utility.
- Environment report script.
- `.gitignore`, basic README, and one passing test.

Acceptance:

- Environment script reports architecture, Python, memory, PyTorch, and accelerator availability.
- Package imports successfully.
- Test suite passes.

## 2. Bigram baseline

Deliver:

- Character vocabulary.
- Encode/decode functions.
- Train/validation split.
- Batch sampler.
- Bigram language model.
- Cross-entropy objective.
- Text sampling with temperature and top-k.

Acceptance:

- Encode/decode round trip passes.
- Batch shapes are tested.
- Loss decreases on a tiny dataset.
- Model generates a sample from a fixed seed.

## 3. Beeja-3M baseline release

Deliver:

- Token and position embeddings.
- Single-head attention, multi-head attention, MLP, residual blocks, normalization, LM head.
- Causal mask.
- Autoregressive generation.

Acceptance:

- Attention output shapes pass.
- Future-token leakage test passes.
- Tiny-batch overfit test reaches the configured threshold.
- Checkpoint save/load preserves logits within tolerance.
- Measured parameter count is reported and tuned close to 3M.
- The saved checkpoint and model card use the `Beeja-3M` name.

## 4. Custom byte-level BPE tokenizer

Deliver:

- Pair counting and merge learning.
- Byte-level base vocabulary.
- Deterministic merge rules.
- Encode, decode, save, and load.
- Special-token handling.
- Compression report.

Acceptance:

- Unicode and ASCII round trips pass.
- Saved tokenizer reloads identically.
- Merge determinism test passes.
- Compression is measured on held-out text.

## 5. Intermediate pretraining

Deliver:

- Subword data pipeline.
- Config-driven model and training.
- AdamW, warm-up, cosine decay, clipping, accumulation.
- Validation, checkpointing, resume, and sample generation.
- Colab notebook or script.

Acceptance:

- Smoke config completes locally.
- Interrupted training resumes from the correct step.
- Validation loss is logged without data leakage.
- Full-run command is prepared but not automatically launched.

## 6. Modern architecture

Deliver:

- RoPE.
- RMSNorm.
- SwiGLU.
- Weight tying.
- Mixed precision where supported.
- Optional attention optimization after reference implementation passes.

Acceptance:

- Each component has a focused numerical or shape test.
- Baseline-versus-modern comparison uses the same controlled smoke setup.
- Parameter counts and measured memory are reported.

## 7. Instruction tuning

Deliver:

- Conversation schema.
- Chat template and special tokens.
- Assistant-only loss mask.
- Supervised fine-tuning pipeline.
- Multi-turn generation and stopping.

Acceptance:

- Prompt tokens are excluded from assistant-only loss as configured.
- Chat formatting round trip is tested.
- Small synthetic dataset can be overfit for pipeline validation.

## 8. Evaluation

Deliver:

- Loss/perplexity or bits-per-byte where appropriate.
- Repetition and diversity metrics.
- Prompt-following test set.
- Latency, tokens/second, peak memory, model size.
- Reproducible report generator.

Acceptance:

- Metrics run from a saved checkpoint.
- Results include configuration, seed, hardware, and limitations.
- No LLM-as-judge is the sole metric.

## 9. MLX/local inference

Deliver:

- PyTorch-to-MLX conversion or a justified MLX inference implementation.
- Numerical comparison on a fixed prompt.
- Apple Silicon benchmark.

Acceptance:

- Converted checkpoint produces sufficiently close logits or a documented difference.
- Local memory and speed are measured.

## 10. Application

Deliver:

- Streaming CLI first.
- Optional local web UI after CLI works.
- Model/config selection and safe generation limits.

Acceptance:

- Fresh setup instructions reproduce a chat session.
- Invalid checkpoint and out-of-memory cases fail clearly.

## 11. Polish

Deliver:

- Architecture diagram.
- Final README.
- Reproduction guide.
- Experiment table.
- Limitations and future work.
- Portfolio demo script.

Acceptance:

- Clean install and smoke test work from documented commands.
- Repository tells a coherent learning and engineering story.
