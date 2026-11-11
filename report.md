# Soup take-home · LoRA → merge → 4-bit
Qwen2.5-1.5B-Instruct on text-to-SQL [A]

Toktarkhan Karymbayev · 11 Nov 2026 [I] · [GitHub repository](https://github.com/tokha-t/soup-takehome) · **Verdict: DON'T SHIP** (§4)

**Choices.** Qwen2.5-1.5B-Instruct uses Apache-2.0. The 3B uses Qwen’s research license [I]. The CC BY 4.0 b-mc2/sql-create-context dataset [I] supplies checkable answers that copy identifiers from the prompt and come last, where truncation can cut them. Training used 1,500 rows. The official evaluation retained 192 of 200 held-out rows after removing 8 that shared a CREATE TABLE statement with training [S]. The POST-HOC cohort drops row 2961, whose gold references a missing column, leaving 191 [P]. LoRA used rank 16, α = 32, all 7 linear targets per layer, 2 epochs and learning rate 2e-4, with fp16, gradient checkpointing and one Kaggle T4 [A]. PEFT trained the main runs after Soup rejected the input with “pre_tokenized row must have 'input_ids'.” The same rows passed strict validation as ChatML after training [A].

The preregistration was hashed at 2026-11-10 20:57:25.073 UTC, SHA-256 [d2f3781007f309964cb09b6a5f11a96359d37ae5cd76593c0598a2d5d672211f](https://github.com/tokha-t/soup-takehome/blob/main/logs/stages/training/prereg.log). The amendment was hashed at 2026-11-11 03:15:05 UTC+05, after training and before the main merge or verification output [A].

## 1. Memory (GiB)

| Phase | Predicted | Allocated | Reserved | nvidia-smi | Src |
|---|---:|---:|---:|---:|---|
| Base load, fp16 | 2.90 | 2.94 | 3.02 | 3.10 | [S, M] |
| Training, batch 8, max 1,912 padded tokens | 6.17–7.26 | 5.14 | 10.38 | 10.52 | [M, T] |
| Merge fp16 / fp32 (CPU) | 2.9–5.8 / 5.8–11.5 | — | — | — | [S, M] |
| Quantize to NF4 | 1.1–1.3 | 1.10 | 1.14 | 1.26 | [S, M] |
| NF4 decoding, batch 8 | 1.2–1.5 | 1.44 | 1.53 | 1.67 | [S, M, D] |

The CPU merge peaked at 5.87 / 9.05 GiB process RSS [M]. The model adds weights (2 bytes × 1.54B parameters), adapter, gradients and AdamW (16 bytes × 18.5M trainable parameters), and full-vocabulary logits (10–14 bytes × 151,936 entries, or 1.45–2.03 MiB per padded token) against about 0.17 MiB per token for checkpointed activations [A, S]. Masking the loss leaves full-position logits allocated. The PEFT fallback projected a suffix containing the answer positions. The estimate followed the bake-off OOMs [A, B]. The checkpointed fit over 3 shapes was 3.103 GiB + 1.828 MiB per padded token, against 3.15 GiB + 1.62–2.20 MiB predicted [G].

Allocated measures tensor memory. PyTorch’s reserved peak was 5.24 GiB higher, and the device-wide sampled peak was another 0.14 GiB higher. The 200 ms sampler can miss brief peaks, so use the in-process allocator peaks for PyTorch [M, T]. Soup pre-flight predicted 28.7 GB for the bake-off. The checkpointed SQL PEFT run sampled 10.33 GiB [B, T]. The main PEFT trainer reported 737 non-pad tokens/s including evaluation. Step times after warm-up give 974 non-pad tokens/s excluding evaluation, from one shared-GPU run [T].

## 2. Proof the model trained

Loss can fall as the model learns output format, and a saved artifact can lose an edit. In the bake-off rescore, base strict match was 0%, base lenient match was 52% after SQL extraction, and tuned lenient match was 77% [R]. The shuffled control's first and last step losses were 3.323 and 1.565 [T].

**The check** (`verify.py`) compares weights, behavior and identity. The fp16 merge changed the expected 196 tensors (28 layers × 7 targets). Across modules, the relative error in merged − base versus the scaled B × A was 1.46–5.42% [V]. Behavior uses paired bootstrap answer loss, lenient match and valid SQL. Identity compares the NF4 output distribution with its own tuned fp16 merge and the base. The training label requires a 95% lower bound of at least 10% answer loss reduction and at least 10 points of lenient gain (exact McNemar p < 0.01) [A, V].

| POST-HOC greedy, 191 rows [H] | Answer NLL | Lenient match | Valid SQL | Closer to tuned | Label [V] |
|---|---:|---:|---:|---:|---|
| Base, fp16 | 0.35989 | 39.8% | 94.2% | — | — |
| Base, NF4 | 0.38726 | 29.8% | 84.3% | not measured | — |
| Merged, fp16 | 0.07752 | 70.7% | 95.8% | — | TRAINED |
| Merged, NF4 | 0.10765 | 60.2% | 86.4% | 100% | TRAINED, RETAINED |
| Placebo, lr = 0 | 0.35989 | 39.8% | 94.2% | not measured | NOT_TRAINED |
| Weak, B × 0.01 / × 0.1 | 0.34137 / 0.17709 | 42.4% / 58.1% | 94.8% / 95.3% | not measured | NOT_TRAINED / TRAINED |
| Shuffled, NF4 | 0.71947 | 0.0% | 0.0% | 100% | NOT_TRAINED |

Training gained 30.9 points of lenient match, NF4 cost 10.5, and the net gain was 20.4 [H]. **Official:** S1 loss fell 78.2%, S2 content gain was 0.613 nats/token, S3 retention was 98.7% (lower bound 92.2%), S4 identity was 192/192, and S8 controls met the amended rule [S, V]. The fp16 and NF4 merges agree on 98.44% of teacher-forced next-token choices. At 26.19 mean gold answer tokens, the independence calculation gives 33.74% with at least one differing choice [C]. Complex queries account for 15 of 26 NF4 greedy failures [Dg].

**What the check does not detect.** Relative retention can pass while absolute accuracy drops by 10.5 points [H, V]. The split checks schema overlap, and lenient string match can miss semantically equivalent SQL. The synthetic execution probe used fixtures with many empty results. Generalization beyond Soup’s 8 suites, training-seed variance and other runtimes remain unmeasured [A, D].

## 3. Silent failures

| Failure | How checked | data doctor | Pre-flight | soup ship | Evidence |
|---|---|---|---|---|---|
| Answers cut in 3.47% of rows | Full scan, p96 limit | Truncation check OK | — | — | [D] |
| Wrong gold and 6 schemas that fail to build | SQLite gold check | — | — | — | [P] |
| Task system prompt missing | Prompt ablation | — | — | User-only task messages | [Dg, C] |
| Dead or weak adapter | Placebo and B × 0.01 | — | lr = 0 bench refused | Placebo and weak both DON'T SHIP | [D, C] |
| Learns format only | Shuffled answers | — | — | DON'T SHIP | [V, C] |
| NF4 validity drops | 18 vs 0 flips, p = 7.6e-6 | — | — | SHIP | [F, C] |
| SQLite checker rejects 5.2% of gold | Gold calibration | — | — | — | [P] |
| 8-bit answers depend on batching | 1 of 10 differs | — | — | — | [F] |
| Wrong file shipped | Identity test | — | — | `--adapter` scores a live adapter | [D, C] |

The repetition-penalty prediction missed. In the isolated greedy test, moving from 1.0 to 1.1 raised lenient match by 4.69 points [A, D]. The leak prediction missed too. Match was 57.0% on the deliberately overlapping cohort versus 59.9% on the disjoint cohort, with different category mixes [A, D].

## 4. Verdict: DON'T SHIP

**Official.** Seven of eight preregistered criteria passed [S]. S5 requires at least 99% stopping and 95% valid SQL under the shipped settings. Stopping passed. The validator accepted 182/192 gold answers (94.8%), below the 99% calibration floor. S5 was unmeasured and counts as a miss [S, P].

**POST-HOC**, the repaired validator accepts 191/192 gold answers [P]. Shipped NF4 reaches 87.1% valid SQL (499/573) and 99.3% stopping (569/573) on the repaired cohort [P, H]. Soup’s separate gate scored the portable merged NF4 file. Strict task match rose from 0% to 66% and no suite fell more than 0.05, so Soup returned SHIP [C]. Its CLI sends user-only task messages. The greedy ablation with Qwen’s default system message gave 77.0% validity [C, Dg].

**What must change in the 4-bit file.** No tested NF4 decoding condition reaches 95% [Dg]. Greedy fp16 reaches 95.8%, and greedy 8-bit reaches 95.3–95.8% depending on SQLite [Dg, F]. Try another quantization method and require the exact file to pass S5, preserve the task system prompt (9.4-point loss without it, p = 0.0014), keep the repaired validator, and fix or drop row 2961 [F, P]. Our NF4 weight file is byte-identical to the repeated `soup merge --save-format 4bit` export [C].

**Not done:** completed `quant-check`, completed `--noise-floor`, fp32-to-NF4, lr = 0 bench, base-model token agreement, GGUF and multiple training seeds [A, C, D].

**What I would change in Soup.** Add system messages and a gold-calibrated executable scorer to tasks. Score the quantized file against an absolute bar with paired tests. Today leg 1 accepts any strict gain, and 0.05 on a 40-item suite is 2 items [C]. Make truncation a full-file scan in `data doctor` [D].

**Sources** · [A] prereg.md, amendment, configs/train_peft.py and timestamped prereg logs · [B] logs/bakeoff/ · [R] results/rescore.json · [P] results/s5_posthoc.json · [Dg] results/s5_diagnosis.json · [F] results/s5_followup.json · [V] results/verify_*.json and probe_P5.json · [M] results/memory_phases.json · [G] results/memory_grid.json · [S] results/summary.json and amendment_split.json · [H] results/report_headline_191.json · [C] results/consistency.json (Soup live_eval.py line 172) · [D] results/probe_P*.json and diagnostic logs · [T] results/report_evidence.json and train_*.json · [I] repository README.
