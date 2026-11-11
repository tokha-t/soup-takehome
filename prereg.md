# Main SQL pipeline pre-registration

## Memory prediction

Batch 8, rank 16, maximum length 256. Decimal GB. Device estimates exclude the separately listed CUDA context and use full-vocabulary logits as specified.

| Term or phase | Prediction |
| --- | --- |
| Parameters | 1,543,714,304 |
| Base weights fp16 | 3.087 GB |
| LoRA parameters | 18,464,768 |
| Adapter, gradients and AdamW fp32 | 0.295 GB |
| Logits per token | 1.519–2.127 MB |
| Activations per token, checkpointing on | 0.183 MB |
| Activations per token, checkpointing off | 2.710 MB |
| CUDA context visible to NVIDIA sampler | 0.3–0.6 GB |
| NF4 checkpoint | 1.143 GB |
| Training peak, checkpointing on | 6.869–8.114 GB |
| Training peak, checkpointing off | 12.044–13.288 GB |
| model load fp16 | 3.087 GB |
| adapter save | 3.161 GB |
| merge fp32 | 6.359 GB |
| merge fp16 | 3.244 GB |
| merge bf16 | 3.244 GB |
| quantize merged | 1.171 GB |
| quantize base | 1.171 GB |
| inference 4bit | 1.207 GB |

The optimized PEFT fallback projects only a suffix covering answer positions. Its measured logits cost may be lower than the full-vocabulary estimate.

## Probe predictions

| ID | Soup gate run | Predicted |
|---|---|---|
| P1 | `data doctor` | Says OK while about 4% of rows lose answer tokens |
| P2 | `data doctor` | Uncertain: depends on whether its collator masks pads by token id |
| P3 | none | Several points lower match |
| P4 | none | Lower match with repetition penalty 1.1, because copying schema column names gets penalized |
| P5 | none | bf16 merge rounds away a measurable share of edited elements and raises answer loss slightly |
| P6 | `soup ship --adapter` | SHIP (blind to the file); `--tuned` on the real file catches it via leg 1 |
| P7 | `adapters check`, `ship --adapter` | Both catch it |
| P8 | `adapters check` | Says alive (miss); ship leg 1 fails it |
| P9 | none | A few skipped steps early, no NaN |
| P10 | none | Higher match on leaked rows |

## SHIP criteria

SHIP only if all hold. S1 is measured on merged_fp16; S2–S7 on merged_4bit.

- **S1 trained**: merged_fp16 answer loss below base; bootstrap 95% interval excludes 0.
- **S2 content, not format**: content gain (Part C) interval above 0.
- **S3 retention**: retention ≥ 90% with its 95% lower bound ≥ 80%; lenient match beats
  base (fp16) by at least 10 points (exact McNemar p < 0.01).
- **S4 identity**: at least 95% of eval rows are closer to merged than to base (B3).
- **S5 stops and parses**: under the shipped settings, ≥ 99% of generations end with the
  stop token and ≥ 95% are valid SQL.
- **S6 no forgetting**: soup ship (b), leg 2: no general suite drops more than 0.05.
  A fail is reported with the number of items behind it.
- **S7 deployable as-is**: loads standalone with tokenizer and chat template; the shipped
  generation_config costs at most 2 points of lenient match versus greedy, averaged
  over 3 sampling seeds.
- **S8 controls**: placebo flagged NOT_TRAINED, shuffled run TRAINED_NOT_ON_TASK. If either
  control gets any other label, the check is broken and nothing ships.

Any miss means DON'T SHIP, with the exact change that would fix it.

**Definitions used when applying these (part of the pre-registration):**
- S2 on merged_4bit means content gain = loss(shuffled_4bit) − loss(merged_4bit), with both
  models quantized by the identical procedure (Step 7).
- S5 "under the shipped settings" means pooled over the same 3 sampling seeds as S7
  (600 generations).
- The valid-SQL check is calibrated on the gold answers first (Step 1); S5's 95% applies only
  if gold scores at least 99% on the same check.

## Analysis choices fixed before training

Use paired row bootstrap with 10,000 resamples (seed 42). Compute token-weighted NLL within each sample and percentile 95% intervals. Treat KL at or below 1e-6 nats per answer token as approximately zero. Freeze source and model revisions, split hashes, training hyperparameters, generation cap, and sampling seeds 42, 43 and 44.

Valid SQL means SQLite accepts EXPLAIN QUERY PLAN for the extracted SQL. Also report raw-output validity to distinguish format from schema errors.

## Split hashes

- train 3a2ec25929442e9e1bd955ad435429f0539cec6cc9643202dcc3bc0bb2be3c17

- train_shuffled e42c87e6013e045690ced56ec03856cbe67e8c173cb54a890fe8ad7c240b5045

- eval 75237ff27b2c0adf4cf70d8d536668ffd78de126e6ec189c144ddc69dbd1e7b2

- eval_leaky b4234874d5f19cf12a3cdfb7ee0b832d6cc3c8bb56cc3b19d677c3816d2b9c1d
