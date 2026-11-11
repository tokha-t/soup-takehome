# Evidence notes and limits

These notes clarify the completed experiment. They do not alter the archived results, preregistrations, hashes, raw logs, or the DON'T SHIP verdict. The original published implementation is preserved at Git commit `f358fa5`. Reproduction changes are documented separately in [REPRODUCE.md](REPRODUCE.md).

## Original and amended decisions

The original registration was recorded at 01:57:25 on 11 November 2026, Asia/Almaty, before the real training command at 01:59:02. The amendment was recorded at 03:15:05, after training and its periodic held-out losses, and before the main merge and artifact verification. Therefore, seven of eight **amended** criteria passed; the amended rules were not fully blind to training outcomes.

The amendment changed the classifier to compare every candidate with the base under the same minimum-effect rules. This avoids using the shuffled candidate as its own reference. The original registration and amendment remain separate, unchanged records.

The official amended evaluation has 192 rows after excluding eight with individual CREATE TABLE overlap. The POST-HOC table removes the broken gold query in row 2961 as well, leaving 191. The three shipped-setting seeds therefore yield 573 post-hoc generations. Compatibility scoring applies the same documented schema/dialect rules to gold and generated SQL; it measures planning acceptance, not semantic correctness.

## Memory predictions and measurements

The original registration uses decimal GB. The table converts the unrounded [prediction JSON](results/memory_prediction.json) to GiB (bytes / 2^30); the original decimal figures below are rounded as printed in [prereg.md](prereg.md). Predictions exclude the separately budgeted CUDA context. The later column comes from the [post-training amendment](prereg_amendment_1.md), which preceded the recorded export measurements. These are distinct records, not retrospectively substituted predictions.

| Phase / domain | Original before-training prediction | Measured peak (GiB) | Later amendment (GiB) |
|---|---|---:|---|
| Base load, GPU | 3.087 GB = 2.8754 GiB | 2.9444 allocated | 2.9 |
| Checkpointed training, GPU | 6.869–8.114 GB = 6.3971–7.5563 GiB | 5.1407 allocated | 6.40–7.60 at 2,048 tokens |
| fp16 / fp32 merge, GPU | 3.244 / 6.359 GB = 3.0211 / 5.9221 GiB | Not run on GPU | 3.0 / 5.8 if on CUDA |
| fp16 / fp32 merge, CPU RAM | Not estimated for CPU RSS | 5.8698 / 9.0470 RSS | 2.9–5.8 / 5.8–11.5 RSS |
| NF4 export, GPU | 1.171 GB = 1.0902 GiB | 1.0990 allocated | 1.1–1.3 |
| NF4 export, CPU RAM | Not estimated | 6.3649 RSS | Approximately 2.9 RSS |
| NF4 decoding, GPU | 1.207 GB = 1.1238 GiB | 1.4377 allocated | 1.2–1.5 |

Original merge predictions include a temporary largest-matrix term ([calculation](predict_memory.py)); they are device tensor estimates, not CPU process-RSS budgets. The actual merges were on CPU, so comparing those original GPU estimates directly with measured CPU RSS would mix memory domains. The base-only load estimate also omits the adapter initialized during the measured load: its 0.0688 GiB explains almost all the 2.8754 → 2.9444 difference.

Both tabulated training predictions use batch 8 × length 256 = 2,048 padded tokens. The amendment is 3.2 GiB + tokens × 1.6–2.2 MiB. The observed maximum was 1,912 padded tokens, giving 6.1875–7.3078 GiB under that formula. The PEFT path projects only a suffix covering answer positions, reducing logits memory compared with the original full-position estimate. The measured training phase includes periodic evaluation; it is not an isolated training-step peak.

A separate diagnostic in [report_evidence.json](results/report_evidence.json) gives 6.1748–7.2578 GiB at 1,912 tokens. Despite its `prediction_source` label, those values use the grid's predicted intercept (3.15) and slopes (1.62–2.20), not its fitted values. The [memory grid](results/memory_grid.json) fits 3.102746 GiB + 1.828026 MiB/token over three checkpointed shapes. Neither diagnostic is presented as the original pre-training prediction. The archived JSON remains unchanged.

| Phase | GPU allocated | GPU reserved | NVIDIA sampled | CPU RSS |
|---|---:|---:|---:|---:|
| Training model load | 2.9444 | 3.02 | 3.10 | Not compared |
| Training including periodic eval | 5.1407 | 10.3789 | 10.5186 | Not compared |
| NF4 export | 1.0990 | 1.1445 | 1.2627 | 6.3649 |
| NF4 decoding | 1.4377 | 1.5332 | 1.6650 | Not compared |

All values in this second table are GiB. Sources: [training phases](results/memory_phases_training.json), [CPU merges](results/memory_phases_2a.json), [NF4 export](results/memory_phases_3.json), [decoding](results/memory_phases_5.json), and [raw GPU samples](logs/nvidia_smi.csv). Use `real_model_load` from the training stage: the identically named verification phase has other models resident. Successful merge attempts are used; failed attempts remain archived.

The combined CPU merge/CUDA NF4 export phase exceeded the amended approximately 2.9 GiB host estimate. That estimate did not adequately cover the implemented phase's live CPU model/tensor and runtime footprint; the archived sampling does not isolate each component.

Allocated and reserved are PyTorch allocator measurements. Device-wide nvidia-smi samples include a broader scope but can miss transients between 200 ms observations. Differences between separately occurring phase maxima do not isolate the exact CUDA-context allocation.

The Soup profiles in `results/memory_grid.json` use q_proj/v_proj targets (2,179,072 trainable parameters), while the measured grid uses all seven targets (18,464,768). These are not identical adapter configurations. Treat the comparison as diagnostic, not as a controlled estimator-accuracy test. The bake-off Soup and PEFT paths also differ.

## Training-console capture

The archived `peft_real_attempt1.log`, `peft_placebo_attempt1.log`, and `peft_shuffled_attempt1.log` retain timestamped command launch and exit status only. Raw per-step console output is not present in this submission.

Per-step metrics and durations survive in `results/train_*.json`; timestamped GPU samples and phase snapshots also survive. They are separate evidence sources, not replacements relabeled as original console logs. No synthetic raw training log has been reconstructed.

## Meaning of control and retention labels

The shuffled-target run is a mismatched-target negative control. It can teach incompatible identifiers as well as output patterns. Its worse held-out loss and zero validity do not establish pure format learning. The historical `format_gain` and `content_gain` fields retain their original names for reproducibility; interpret the latter as the measured advantage over shuffled-target training, not a clean causal decomposition of format and content.

The amended NOT_TRAINED label means that a candidate failed the chosen minimum evidence for meaningful task improvement. It does not prove that no optimizer steps occurred.

The retention statistic is:

`(NLL(base NF4) − NLL(candidate NF4)) / (NLL(base fp16) − NLL(candidate fp16))`.

For the real model, the denominator is positive, and the reported 98.7% measures preservation of a beneficial answer-loss improvement. A ratio of two negative effects can also be positive: the historical checker labels the shuffled export RETAINED at approximately 97.1%, meaning that its harmful change persisted. That is not evidence of useful learning or release readiness. The archived classifier and outputs are preserved; only interpret retention as preservation of a benefit when the original effect is positive.

The identity check asks whether an export resembles its own tuned reference. It can correctly identify a harmful shuffled model as well as a useful model. Identity, useful learning, and suitability for release are different checks.

## Silent-failure coverage

“Not tested” below means no targeted test established that gate's response to this failure. A gate running elsewhere is not evidence that it catches every failure. “Not applicable” identifies a check whose input/scope does not include the failing component.

| Failure | data doctor | Pre-flight | soup ship |
|---|---|---|---|
| Answer/stop truncation | Tested and missed as a warning/failure gate: count reported, truncation verdict OK | Not tested for this failure | Not tested for this failure |
| Wrong gold / malformed evaluation schemas | Not tested on this faulty evaluation cohort | Not tested for this failure | Not tested as a dedicated gold-validity/calibration check |
| External checker rejects valid gold | Not applicable: external checker | Not applicable: external checker | Not applicable: Soup used a different scorer |
| Missing task system prompt | Not tested as a train/inference mismatch | Not tested for this failure | Source reviewed; not tested as a dedicated live rejection case |
| Dead placebo adapter | Not applicable to a saved adapter | Not tested as dead-adapter detection | Tested and caught: adapter_noop |
| Weak B × 0.01 adapter | Not applicable to a saved adapter | Not tested | Tested and caught: task-win failure |
| Mismatched-target training | Not tested on shuffled targets | Not tested | Tested and caught: task-win failure and five suite regressions |
| NF4 absolute validity loss | Not applicable to exported-model behavior | Not tested for this export failure | Tested and missed relative to our absolute bar: Soup returned SHIP under its own rule |
| Wrong NF4 artifact | Not applicable to artifact identity | Not tested | Tested and rejected for task-win failure, not a dedicated identity check |

- **Truncation:** [our scan](results/probe_P1.json) found 52/1,500 rows losing answer/stop tokens (3.47%). [Soup's result](results/probe_P1_soup.json) scanned all 1,500 rows, reported 58 truncated rows (3.9%), yet marked `truncation_risk` as `OK` at max_length 147. The overall doctor result was `MINOR` for a separate generation-marker issue. These counts measure different aspects of truncation. Soup noticed truncation; it did not elevate this truncation finding into a warning/failure. [Raw command/output](logs/stages/7/probe_P1_doctor.log). The proposed improvement is an explicit answer/EOS-loss check with an actionable threshold, not just increasing the sample size.
- **Gold/checker:** [POST-HOC calibration](results/s5_posthoc.json) records six original schema-build failures and 182/192 accepted gold queries. Compatibility repairs raise this to 191/192; row 2961 remains invalid. This does not demonstrate that a Soup gate was tested against all those defects.
- **Prompt:** [source inspection](results/consistency.json) (`soup_ship_prompt_source`) establishes user-only task rendering. The [prompt ablation](results/probe_P3.json) and [paired post-hoc scoring](results/s5_followup.json) measure the 9.4-point validity loss and p = 0.0014. That loss is our measurement, not an SQL-validity result emitted by Soup.
- **Dead/weak/shuffled:** [placebo ship](results/ship_c.json), [weak ship](results/ship_weak_001.json), [shuffled ship](results/ship_shuffled.json). The [adapter checks](results/soup_adapter_checks.json) call placebo inactive but weak/shuffled alive: activity is not evidence of useful task learning.
- **Pre-flight qualification:** the [lr=0 bench attempt](logs/stages/7/bench_placebo.log) failed configuration validation (`training -> lr: Input should be greater than 0`). The benchmark never ran. This is not a measured pre-flight detection of a dead saved adapter.
- **NF4 quality:** [Soup's SHIP](results/ship_b.json), [18 validity regressions versus zero gains](results/s5_followup.json), and [87.1% shipped-setting validity](results/s5_posthoc.json) describe different gates applied to the same tested export. Soup's relative task rule does not enforce this project's absolute 95% SQL-validity requirement.
- **Wrong artifact:** [ship --tuned rejection](results/ship_d.json), [independent identity check](results/verify_wrong.json), and [probe P6](results/probe_P6.json). Rejection on task performance does not establish a dedicated artifact-identity check in Soup.

## Proposed release experiment — not run

Repeating the existing NF4 export is not a remedy: [the repeat audit](results/consistency.json) found identical saved weights and greedy outputs. The next proposed change is **calibration-based GPTQ INT4, group size 128**, starting from the existing fp16 merge. GPTQ uses calibration data and a different weight-error optimization procedure; Transformers supports custom calibration text through GPT-QModel. This is a hypothesis to test, not a claim that GPTQ will preserve SQL identifiers or meet the release bar. [Hugging Face GPTQ documentation](https://huggingface.co/docs/transformers/en/quantization/gptq).

Before executing the experiment, freeze its configuration and software versions in a new dated record; do not edit the original preregistrations. Select 128 complete, non-truncated conversations from the 1,500 training rows with a fixed seed of 42, preserving the task system prompt and chat template. Record the selected source IDs and hash. Use no evaluation rows for calibration. Apply the same calibration procedure to the base and shuffled controls for precision-matched comparisons. Record the quantizer, backend, weights, tokenizer, template and generation configuration hashes. T4 memory fit and compatibility with the chosen loader/Soup path remain untested.

Reload the saved standalone package in the intended inference runtime and rerun the existing checks. Keep the amended thresholds fixed:

| Gate | Requirement for the candidate package |
|---|---|
| S1 | fp16 merge: lower 95% bound of relative answer-NLL improvement at least 10% |
| S2 | Advantage over the identically quantized shuffled control; interval above zero |
| S3 | At least 90% loss-improvement retention, lower 95% bound at least 80%; lenient gain at least 10 percentage points over fp16 base, exact McNemar p < 0.01 |
| S4 | At least 95% of rows closer to the tuned reference than the base |
| S5 | Gold calibration at least 99%; then at least 99% stopping and 95% valid SQL pooled under saved shipped settings |
| S6 | Exact candidate package: no general suite drops more than 0.05 in Soup; inability to load/measure counts as a miss |
| S7 | Standalone load with tokenizer/template; saved generation configuration loses at most 2 match points versus greedy over seeds 42, 43 and 44 |
| S8 | Real TRAINED; placebo and weak_001 NOT_TRAINED; shuffled not TRAINED |

For comparability, use the repaired compatibility scorer, fixed 191-row POST-HOC cohort and seeds 42, 43 and 44 (573 generations for S5/S7); exclude the known broken row 2961 consistently before all comparisons. This would be a new post-hoc follow-up, not a replacement for the official 192-row amended result. Do not tune calibration, decoder or thresholds on these evaluation outcomes. Preserve failed candidates and all raw logs. Because this cohort has already informed diagnosis, an independent schema-disjoint confirmation set is also needed before a real user release.

If GPTQ fails or cannot run through the required checks, the verdict stays DON'T SHIP. fp16 and 8-bit results support investigating quantization loss but do not satisfy the assignment's 4-bit export requirement. Any further quantizer/configuration change should be a separately recorded experiment.

## Remaining uncertainty

SQL planning acceptance does not prove that a query answers the question correctly. String match can reject equivalent SQL and can overlook semantic differences removed by normalization. The synthetic execution probe has many empty results and is not a strong substitute for a realistic execution benchmark. Training uses one seed, the general suites are small, and the export was not tested across deployment runtimes. None of these limits is resolved by the existing SHIP or RETAINED labels.
