# Beeja-3M — Model Card

**Status:** architecture materialised, **untrained** (random initialisation).
Full pretraining is a later stage; no quality metrics are claimed here.

## Identity
- Family: `Beeja`
- Release: `Beeja-3M`
- Type: decoder-only autoregressive Transformer (causal next-token prediction)

## Architecture
| field | value |
|---|---|
| vocab_size | 38 (character-level for this stage) |
| block_size (context) | 128 |
| n_layer | 4 |
| n_head | 4 |
| n_embd | 256 (head_size 64) |
| dropout | 0.0 |
| normalization | pre-norm LayerNorm |
| activation | GELU (4x MLP) |

## Measured parameters
| component | count |
|---|---|
| embedding | 42,496 |
| attention | 1,052,672 |
| mlp | 2,102,272 |
| lm_head | 9,728 |
| norm/other | 4,608 |
| **total** | **3,211,776** |

Parameter memory (fp32): 12.252 MiB. Training memory is larger:
add gradients (~1x), AdamW state (~2x), and activations (∝ batch × context × depth).

## Note on sizing
`architecture.md` lists d=128 as a starting point (~0.8M params). To make the
`Beeja-3M` name honest, `n_embd` was tuned to 256, giving ~3.2M measured params.
The character vocab is tiny, so embeddings are a negligible share; capacity lives
in the attention and MLP blocks.
