# Soup take-home · LoRA → merge → 4-bit
Qwen2.5-1.5B-Instruct on text-to-SQL

Toktarkhan Karymbayev · 11 Nov 2026 · [Evidence repository](https://github.com/tokha-t/soup-takehome) · **Verdict: DON'T SHIP**

**Choices.** Qwen2.5-1.5B-Instruct (Apache-2.0) and the CC BY 4.0 sql-create-context dataset provide a small model and checkable outputs sensitive to identifier copying and truncation [I]. Training used 1,500 rows, rank 16, α = 32, all 7 linear targets per layer, 2 epochs, lr 2e-4, fp16 and checkpointing on one Kaggle T4 [A]. PEFT trained the main runs after Soup rejected the input: “pre_tokenized row must have 'input_ids'.” The same rows later passed strict ChatML validation [A]. Eight overlapping CREATE TABLE rows were excluded after training, leaving 192 for the amended evaluation. POST-HOC comparisons also exclude row 2961's broken gold query, leaving 191 [S, P].

Original criteria were [hashed before training](https://github.com/tokha-t/soup-takehome/blob/main/logs/stages/training/prereg.log), at 01:57:25 on 11 Nov, Asia/Almaty. The amendment at 03:15:05 followed training and its held-out losses, but preceded main merges and artifact verification. Decisions below use the amended rules [A].

## 1. Memory (GiB) [A, M, E]

| Phase / memory domain | Original prediction | Measured peak | Later amendment |
|---|---:|---:|---:|
| Load, GPU | 2.88 | 2.94 | 2.90 |
| Training incl. eval, GPU | 6.40–7.56 | 5.14 | 6.40–7.60 |
| Merge fp16 / fp32, GPU | 3.02 / 5.92 | Not run on GPU | 3.0 / 5.8 |
| Merge fp16 / fp32, CPU | Not estimated | 5.87 / 9.05 | 2.9–5.8 / 5.8–11.5 |
| NF4 export, GPU | 1.09 | 1.10 | 1.1–1.3 |
| NF4 export, CPU | Not estimated | 6.36 | ≈2.9 |
| NF4 decoding, GPU | 1.12 | 1.44 | 1.2–1.5 |

Original estimates precede training; the amendment follows training and precedes these exports. GPU measurements are PyTorch allocated VRAM; CPU measurements are process RSS. The original merge estimates describe device tensors, not CPU RSS. The export's 6.36 GiB host peak exceeded the later 2.9 GiB budget because the implemented phase also merged and loaded CPU tensors [E].

The original calculation adds weights (2 bytes × 1.54B), adapter/gradients/AdamW (16 bytes × 18.5M), logits (1.45–2.03 MiB/padded token) and checkpointed activations (≈0.17 MiB/token). Both training estimates use 8 × 256 = 2,048 tokens; the observed maximum was 1,912. PEFT projects an answer-covering suffix, reducing logits memory. Load also initializes 0.069 GiB of adapters omitted from the original base-only estimate. The full-logit grid fit was 3.103 GiB + 1.828 MiB/token over 3 shapes [G, T].

Training peaked at 5.14 allocated, 10.38 reserved and 10.52 device-wide GiB. These have different scopes; 200 ms NVIDIA sampling can miss peaks. Throughput was 737 non-pad tokens/s including evaluation versus 974 excluding warm-up/evaluation [T]. Raw PEFT logs retain launch/exit only; step JSON and GPU traces survive [E].

## 2. Proof the model trained

Loss can fall without useful task learning, and export can lose an edit. In the bake-off rescore, base strict match was 0%, base lenient match 52% after SQL extraction, and tuned lenient match 77% [R]. The shuffled control's step loss fell from 3.323 to 1.565 while its held-out behavior worsened [T, V].

**The check** (`verify.py`) compares weights, behavior and identity. The fp16 merge changed the expected 196 tensors; relative reconstruction error against scaled B × A was 1.46–5.42% across modules [V]. TRAINED requires a 95% lower bound of at least 10% answer-loss reduction and at least 10 points of lenient gain (exact McNemar p < 0.01), against the base [A]. NOT_TRAINED means insufficient meaningful task improvement, not proof that no optimizer ran.

| POST-HOC greedy, 191 rows [H] | Answer NLL | Lenient match | Valid SQL | Closer to tuned | Label [V] |
|---|---:|---:|---:|---:|---|
| Base, fp16 | 0.35989 | 39.8% | 94.2% | n/a | n/a |
| Base, NF4 | 0.38726 | 29.8% | 84.3% | not measured | n/a |
| Merged, fp16 | 0.07752 | 70.7% | 95.8% | n/a | TRAINED |
| Merged, NF4 | 0.10765 | 60.2% | 86.4% | 100% | TRAINED, RETAINED |
| Placebo, lr = 0 | 0.35989 | 39.8% | 94.2% | not measured | NOT_TRAINED |
| Weak, B × 0.01 / × 0.1 | 0.34137 / 0.17709 | 42.4% / 58.1% | 94.8% / 95.3% | not measured | NOT_TRAINED / TRAINED |
| Shuffled, NF4 | 0.71947 | 0.0% | 0.0% | 100% | NOT_TRAINED |

n/a = not applicable. Training gained 30.9 match points; NF4 cost 10.5, leaving a net 20.4 [H]. **Amended evaluation:** S1 loss fell 78.2%; S2 advantage over mismatched-target training was 0.613 nats/token; S3 retained 98.7% of answer-loss improvement over precision-matched baselines (lower bound 92.2%); S4 identity was 192/192; S8 controls passed [S, V]. Shuffling is not a pure format/content decomposition. Identity and retention can also preserve a harmful change [E].

**Limits.** Relative retention does not establish absolute quality. SQL validity uses documented schema/dialect compatibility repairs and tests planning, not correctness. String match misses equivalent SQL; synthetic execution fixtures often yield empty results. One training seed, small general suites, and one runtime limit generalization [D, E].

## 3. Silent failures

| Failure / own check | Soup check | Observed outcome | Src |
|---|---|---|---|
| Answer/stop tokens cut: 52/1,500 | data doctor: tested | Reports 58 truncated rows but rates truncation OK: missed as a gate | [D] |
| Gold/schema defects; checker calibration | Not tested in Soup | Own gold acceptance: 182/192; repaired: 191/192 | [P] |
| Missing task system prompt | ship: source reviewed | User-only path; own ablation loses 9.4 validity points | [F, C] |
| Dead / weak adapter | ship: tested | Caught: both DON'T SHIP | [D, C] |
| Mismatched-target training | ship: tested | Caught: DON'T SHIP | [V, C] |
| NF4 validity: 18 regressions, 0 gains | ship: tested | SHIP; missed our absolute quality failure | [F, C] |
| Wrong NF4; identity 0/192 | ship --tuned: tested | Rejected for no task gain, not identity | [D, C] |

Unlisted gate/failure pairs were not tested or are not applicable; [E] gives the full matrix. No pre-flight detection claim: lr = 0 was refused at config validation, before its bench ran. Source inspection and our prompt ablation are not a live Soup rejection test.

Predictions failed too: isolated greedy repetition penalty 1.0 → 1.1 raised lenient match by 4.69 points. The deliberately overlapping cohort scored 57.0% versus 59.9% disjoint, with different category mixes [A, D].

## 4. Verdict: DON'T SHIP

**Amended evaluation.** Seven of eight amended criteria passed. S5 requires at least 99% stopping and 95% valid SQL under shipped settings. Stopping passed, but the validator accepted only 182/192 gold queries (94.8%), below its 99% calibration floor. S5 was UNMEASURED and counted as a miss [S, P].

**POST-HOC.** The repaired validator accepts 191/192 gold queries. On those 191 rows, shipped NF4 achieves 87.1% validity (499/573) and 99.3% stopping (569/573) [P, H]. Soup evaluated the portable merged NF4 artifact: strict match rose 0% → 66%, with no suite drop over 0.05, so its gate returned SHIP [C]. Its user-only prompt path differs from training; the default-system greedy ablation gives 77.0% validity [C, Dg].

**Next experiment, not a demonstrated fix.** Repeated NF4 export produced identical weights [C]. Test calibration-based GPTQ INT4 (group size 128) from the fp16 merge, using 128 training-only SQL conversations and no evaluation rows [E]. Preserve the task system prompt; reload and hash the saved package. It must pass the fixed amended S1–S8 rules, with S5's ≥99% stopping and ≥95% valid SQL under shipped settings and ≥99% gold calibration. Use the repaired scorer and the same 191-row POST-HOC cohort/seeds; exclude broken row 2961. No threshold relaxation. No tested NF4 decoder reaches 95%; fp16 and 8-bit results motivate this experiment, not a claim that INT4 will pass [Dg, F, E].

**Not completed:** `quant-check`, `--noise-floor`, fp32-to-NF4, lr = 0 bench, base token agreement, GGUF and multiple training seeds [C, D]. **Soup improvements:** preserve task system messages; enforce absolute quality and gold calibration; warn/fail on lost answer or stop tokens. Leg 1 accepts any strict gain; 0.05 on a 40-item suite is two items [C, D].

**Sources** · [A] prereg.md, amendment, training config/logs · [R] results/rescore.json · [P] results/s5_posthoc.json · [Dg] results/s5_diagnosis.json · [F] results/s5_followup.json · [V] results/verify_*.json, probe_P5.json · [M] results/memory_phases.json · [G] results/memory_grid.json · [S] results/summary.json, amendment_split.json · [H] results/report_headline_191.json · [C] results/ship_*.json, consistency.json · [D] results/probe_P*.json, diagnostic logs · [T] results/report_evidence.json, train_*.json · [E] [EVIDENCE_NOTES.md](https://github.com/tokha-t/soup-takehome/blob/main/EVIDENCE_NOTES.md) · [I] README. Reproduction and AI-use details are linked in README.
