Written 2026-11-11T03:15:05+05:00, after R1–R3 finished training and before any merge, quantization or verification output existed.

# Pre-registration amendment 1

The earlier merge notebook was canceled. No committed merge, quantization or verification output was available when this amendment was written.

The memory formulas were written after the bake-off's out-of-memory errors were known.

**Any SHIP criterion not measured by 10:00 counts as a miss.** At 08:30 Asia/Almaty stop work outside stages 1–6.

## Effective rules for the completed training

- The original hashed pre-registration, training data, and R1–R3 adapters are frozen. The original document remains the record of what was decided before training.
- Padded tokens logged per step and separate eval peak measurement are not applied: training had already finished. Existing padded-token counts are historical evidence if present. A newly instrumented bench is a separate measurement.
- Individual CREATE TABLE overlap was 8. Exclude all 8 affected evaluation rows, keeping 192/200. Source indices excluded: 1950, 1603, 1904, 3479, 2759, 2880, 2867, 3473. The clean evaluation and task files are eval_amended.jsonl and ship_task_amended.jsonl. No replacement evaluation rows and no retraining.
- Every leaky evaluation row shares an individual CREATE TABLE statement with train. Keep this deliberately overlapping cohort only for P10.
- Apply labels against the base only using the fp16 merge for the candidate and the paired greedy outputs. A zero changed tensor count or a lower 95% bound below 10% on relative answer loss reduction gives NOT_TRAINED. With MIN_LOSS met, a gain below 10 points or McNemar p ≥ 0.01 gives TRAINED_NOT_ON_TASK. Meeting both gives TRAINED. Part C is measured only for the real run and has no role in training labels.
- S1 additionally requires MIN_LOSS. S6 must be measured on the real 4-bit model via Soup. S8 requires real TRAINED, placebo and weak_001 NOT_TRAINED, and a measured shuffled label other than TRAINED. Weak_01 is descriptive. All other original SHIP thresholds and retained/lost thresholds carry forward.
- The training-shift to quantization-shift ratio uses KL(base ‖ candidate merge) / KL(base ‖ base_4bit) on answer positions. It is reported and flagged below 3, with no effect on the verdict.
- Use the existing 98-token generation cap and seed 42. For the padding check, compare the first ten clean rows' generated token IDs with EOS handling held fixed. Record mismatches and use single-row greedy scoring if the equality assertion fails on a model.

## Incorporated rules in precedence order

### Patch 1
# Patch 1 to main_pipeline_spec.md

Apply on top of the main spec. Where the two conflict, this patch wins.

## 0. Pre-registration handling

- If `prereg.md` is not written yet: include everything in this patch in it.
- If it is already hashed: write `prereg_amendment_1.md` listing every change below and log its
  sha256 with a timestamp **before the first R1 training step**. If R1 has already started,
  write it anyway, mark it "written after training started at <time>", and the report treats
  its additions as decided after the fact.
- Add this line to the pre-registration: "The memory formulas were written after the bake-off's
  out-of-memory errors were known."

## 1. Units and memory predictions (replaces the values in §12)

All memory in GiB (2^30 bytes) and MiB. Convert nvidia-smi's MiB; never mix in GB.

| Term | Value |
|---|---|
| Weights fp16 | 2.875 GiB |
| Adapter + gradients + AdamW, fp32 | 0.275 GiB |
| Logits during training, per token | 1.45–2.03 MiB |
| Activations, checkpointing on, per token | ~0.17 MiB |
| Activations, checkpointing off, per token | ~2.6 MiB |
| Merged file fp16 / bf16 / fp32 | 2.875 / 2.875 / 5.75 GiB |
| 4-bit model (NF4, double quant) | ≈1.06 GiB |

**Per-phase peak predictions (new).** Log which device each phase actually ran on.

| Phase | Predicted peak |
|---|---|
| Load base fp16 | GPU ≈2.9 GiB allocated |
| Training step | GPU ≈3.2 GiB + padded tokens per batch × 1.6–2.2 MiB |
| Eval loop (no gradients, batch 8) | GPU ≈2.9 GiB + tokens × 0.9–1.5 MiB |
| Merge fp16 / bf16 | GPU ≈3.0 GiB if run on CUDA; otherwise process RSS 2.9–5.8 GiB (one or two model copies) |
| Merge fp32 | GPU ≈5.8 GiB, or RSS 5.8–11.5 GiB |
| Quantize to NF4 | GPU ≈1.1–1.3 GiB; RSS up to ≈2.9 GiB while reading the fp16 checkpoint |
| 4-bit inference, batch 8 | GPU ≈1.2–1.5 GiB |

## 2. Logging additions

- Every training step logs **padded** tokens (batch × padded length) as well as non-pad tokens.
  Memory follows padded tokens.
- Eval batch size 8. Reset peak memory stats around every eval and record the eval peak
  separately from the training-step peak.
- Record the sha256 of every artifact file (adapter safetensors, every merged and 4-bit
  safetensors) in `results/summary.json`. `verify.py` logs the hash of every file it loads.
- Log total system RAM and peak process RSS for every phase (Kaggle, not Colab).

## 3. Data split fix (Step 2)

Grouping by the full `context` string is not enough: multi-table rows can share one
`CREATE TABLE` statement across different context strings.
- Split each `context` into individual `CREATE TABLE` statements and normalize each
  (lowercase, collapse whitespace).
- Union-find: rows that share any normalized statement belong to one group. Assign whole groups
  to train or eval.
- Assert and log that train and eval share **zero** normalized statements.
- `eval_leaky.jsonl` rows must each share at least one statement with train.

## 4. Weak-adapter control (new, P0, ~30 min, no training)

This is the control for "a run that finished without meaningfully changing the model".
- Copy the R1 adapter. Multiply every `lora_B` tensor by 0.01 → `runs/weak_001`, and by 0.1 →
  `runs/weak_01`. Merge each in fp16.
- Run `verify.py` Parts A1, A2, B1 and B4 on both.
- Run `soup adapters check` on both.
- Run `soup ship --base <base> --adapter runs/weak_001 --task-eval data/ship_task.jsonl` (P1).
  Record whether leg 1 passes on a tiny item-level difference.

## 5. Verdict rules (replace the verdict block in §8)

Pre-registered constants:
- `MIN_LOSS`: the lower 95% bound of the relative answer-loss reduction versus base is at least 10%.
- `MIN_TASK`: lenient-match gain versus base is at least 10 points, exact McNemar p < 0.01.

Rules, always decided **against base only**, never against another trained run:
- `NOT_TRAINED`: no tensor changed, or `MIN_LOSS` not met.
- `TRAINED_NOT_ON_TASK`: `MIN_LOSS` met, `MIN_TASK` not met.
- `TRAINED`: both met.

Part C (format vs content) stays as a measurement for S2 and is computed for the real run only.
It no longer decides any label, so no control can pass by construction.

Expected labels: real `TRAINED`; placebo and weak_001 `NOT_TRAINED`; shuffled anything except
`TRAINED`; weak_01 reported without a pass/fail (it shows where the check's threshold sits).

## 6. SHIP criteria changes (S13)

- **S1**: add "and `MIN_LOSS` is met".
- **S6**: if `soup ship` cannot load `merged_4bit`, S6 is UNMEASURED and counts as a miss.
- **S8** (replaces the old text): placebo and weak_001 must be `NOT_TRAINED`; shuffled must get
  any label except `TRAINED` (report which); real must be `TRAINED`. Any other outcome means the
  check is broken and nothing ships.

## 7. `verify.py` fixes

- New arguments: `--shuffled-merged`, `--shuffled4`, and a repeatable `--extra name=path` for the
  weak merges.
- **A1**: compare each merged_X with the base cast through the **same** dtype X, so that cast
  rounding is not counted as training. Read tensor by tensor with safetensors `safe_open`;
  never hold two full fp32 models in RAM. Separately report how many base elements change in
  the bf16→fp16 cast (Qwen's published weights are bf16); that count is a finding of its own.
- **Batched generation**: left padding. Before any scoring, assert that batched greedy output
  equals unbatched greedy output on 10 rows, and log the result.
- Add to "What this check does not detect": string matching rewards the dataset's style
  (quoting, aliases), so a correct query written differently scores as wrong. Execution match
  (P11) would fix that.

## 8. Use more of Soup's own gates

- `soup bench train --config soup.yaml` on the real config and on the lr=0 config (P0).
  Expected: the lr=0 run fails its check (parameters identical between first and last step).
  Record its supervised tokens per second and peak memory as a second memory measurement. If the
  memory pre-flight refuses it, log the full message.
- `soup merge`: `--dtype` and `--save-format` are separate flags. Log the **saved** dtype of every
  `merged_*`. If no fp32 save format exists, build the fp32 reference merge with peft and save it
  in fp32 yourself.
- **4-bit path**: if `soup merge --help` confirms that `--save-format 4bit` writes a bitsandbytes
  4-bit merged model, use it for `merged_4bit`, and push the placebo adapter (all `lora_B` zero,
  so merged equals base) through the identical command to get `base_4bit`. Same for
  `shuffled_4bit`. Log the help text.
- `soup data doctor` for P1 with `--sample 1500` (the whole train file), and record the result as
  "documented behaviour confirmed" or "not confirmed".
- Every probe whose Soup column says "none" gets either the nearest gate's actual output or the
  doc page and sentence that shows the gate does not look at it.
- P2: `soup ship --noise-floor 3`.

## 9. Priorities (time)

- **Added to P0**: §4 weak-adapter control, `soup bench train`, the per-phase predictions.
- **Moved to P2**: A3 SVD subspace retention (code flip rate and cosine stay at P1), 6d source
  check, P9 (except the skipped-step count, already logged in Step 5), P10, P11.
- **Scaling grid reduced to 5 points**: checkpointing on (1,256), (4,256), (8,256);
  off (1,256), (2,256).

## 10. Time check

Log in `findings.md` the time each P0 step finished. At 08:30, stop all P1 work and finish P0.

### Patch 2 (takes precedence)
# Patch 2: amendment after training, plus execution order

**Status this patch starts from.** `prereg.md` was hashed before training. R1 (real), R2
(placebo) and R3 (shuffled) finished training with peft, because Soup rejected the data format.
No merge, quantization or verification output exists yet. Patch 1 and this patch both apply;
where they conflict, this one wins.

## 1. Amendment (first, 15 min)

Write `prereg_amendment_1.md` containing every rule from patch 1 and this patch that decides how
results are judged. Its first line:

> Written <timestamp>, after R1–R3 finished training and before any merge, quantization or
> verification output existed.

Log its sha256 with a timestamp. Patch 1 items that could only apply during training (padded
tokens logged per step, eval peak measured during training) are listed as "not applied:
training had already finished". (P2: reconstruct padded tokens per step by replaying the seeded
sampler and collator without running the model.)

Add this rule: **any SHIP criterion not measured by 10:00 counts as a miss.**

## 2. Split check (10 min)

The split was made by whole `context` strings, before patch 1. Count normalized
`CREATE TABLE` statements shared between `train.jsonl` and `eval.jsonl`.
- Zero: log it and continue.
- Not zero: **do not retrain.** Remove the affected eval rows from every evaluation, log how
  many, and record the removal in the amendment before any evaluation runs.

## 3. Soup: what training did not use (30 min box)

- Copy the exact Soup data-format error into `logs/failed/soup_format/` with what was tried.
- Find a format Soup accepts (https://trysoup.dev/docs/data-formats) and write
  `data/train_soup.jsonl`: the same rows, converted. **Do not retrain the main runs.**
- `soup data validate data/train_soup.jsonl --min-valid-fraction 1.0`. By default the command
  only fails when no row is usable; Soup's own `soup ci init` workflow uses 1.0.
- **New probe P12 (P1, 5 min):** copy the file, break half the rows, run `soup data validate`
  with and without `--min-valid-fraction 1.0`, record both exit codes.
- For `soup bench train` (patch 1 §8): `soup init --template chat` writes `quantization: 4bit`;
  set it to `none`, and match R1's settings. Run it on the real and on the lr=0 config.
- Write down for the report: "Trained with peft because Soup rejected the data format
  (<error>). <Fixed by X after training / not fixed within the time box>. All post-training
  gates are Soup's."

## 4. Additions to `verify.py`

- **B1 extra row:** base plus the unmerged R1 adapter (peft, fp16). Expected to match
  `merged_fp16` closely; the difference is what merging changed.
- **Quantization-noise ratio (reported, not a gate):** training shift ÷ quantization shift, with
  training shift = mean per-token KL(P_base ‖ P_merged_fp16) and quantization shift = mean
  per-token KL(P_base ‖ P_base_4bit), both on answer positions. Flag it if below 3. Report it for
  the real run and for weak_001 and weak_01.

## 5. Execution order (SHIP-critical first)

Run every stage as a committed Kaggle version, with the training outputs attached as input, and
download outputs after each stage so a cancelled session loses nothing.

1. Amendment and split check (§1, §2).
2. Merges: R1 in fp16 (production), fp32 (reference) and bf16; placebo, shuffled, weak_001 and
   weak_01 in fp16.
3. 4-bit: `merged_4bit`; `base_4bit` (the placebo through the same path); `shuffled_4bit`.
4. `verify.py` core: B1 (with the unmerged row), B3, B4 greedy, Part C, and verdicts for real,
   placebo, shuffled, weak_001 and weak_01. Covers S1–S4 and S8.
5. P4 sampling generations (3 seeds) and P3. Covers S5 and S7.
6. `soup ship` (b), which covers S6; then (a) and (c).
7. A1/A2 merge dtypes (P5), memory phase peaks, `adapters check` and `audit`, `bench train`,
   `data doctor --sample 1500` (P1), `data validate` (P12).
8. If time remains: memory grid (5 points), `quant-check`, ship (d), ship on weak_001, A3 code
   flip rate.

Cutoff unchanged: at 08:30, stop everything outside stages 1–6.
