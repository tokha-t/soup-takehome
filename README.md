# Soup take-home

Toktarkhan Karymbayev · 11 Nov 2026

The official verdict is **DON'T SHIP**. Seven of eight amended gates passed. S5 counted as a miss because gold SQL failed the required validator calibration. The POST-HOC compatibility check accepts 191 of 192 gold queries. The shipped NF4 decoder produced 499 valid queries in 573 attempts on that cohort, below the 95% bar. [Official summary](results/summary.json) · [POST-HOC S5](results/s5_posthoc.json)

- [Two-page report](report.pdf) · [Report source](report.md) · [Surprise](surprise.md) · [Candidate's AI disclosure](ai_usage.md) · [Preserved agent execution record](ai_usage_agent.md)
- [Evidence notes and measurement limits](EVIDENCE_NOTES.md) · [Reproduction guide](REPRODUCE.md) · [Read-only CPU audit](audit_saved_results.py)
- [Preregistration](prereg.md) · [Pre-training hash log](logs/stages/training/prereg.log#L1) · [Amendment](prereg_amendment_1.md) · [Amendment hash log](logs/prereg_amendment_1.log#L1)
- [Data builder](data/build_data.py) · [Split amendment](apply_amendment_split.py) · [Training code](configs/train_peft.py) · [Configs](configs/) · [Verifier](verify.py) · [Probes](probes/) · [Findings](findings.md)
- [Result JSON](results/) · [Raw timestamped logs](logs/) · [GPU samples with stage labels](logs/nvidia_smi.csv) · [Raw nvidia-smi snapshots](logs/nvidia_smi_raw/)
- [Adapter hashes](results/training_adapter_hashes.json) · [Production artifact hashes](results/artifact_hashes_4.json) · [POST-HOC ship and tensor-hash audit](results/consistency.json)

The repository records model weights by SHA-256. The committed notebook versions document their inputs, code, outputs and execution environments. Stage-local CSVs and unmodified NVIDIA snapshots sit alongside the combined GPU sample table.

The report separates original memory predictions from measured CPU/GPU peaks and the later amendment. The evidence notes provide the complete Soup-check coverage matrix and a proposed, untested GPTQ INT4 follow-up with unchanged release thresholds.

For email review, start with the report and evidence notes. The CPU audit recomputes the central saved-output results without downloading a model. The reproduction guide separately covers checkpoint recovery, standalone GPU verification and the staged Kaggle workflow. Reruns use a disposable copy and a fresh time budget; the original code and evidence remain available at [the initial submission commit](https://github.com/tokha-t/soup-takehome/tree/f358fa5c233aefda296e14768c39400bf9a4ca38).

## Public Kaggle notebooks

| Notebook | Committed version |
|---|---|
| Dataset bake-off | [368173550](https://www.kaggle.com/code/toktarkhan/soup-dataset-bake-off?scriptVersionId=368173550) |
| SQL rescore | [368194383](https://www.kaggle.com/code/toktarkhan/soup-take-home-sql-rescore?scriptVersionId=368194383) |
| SQL rescore evidence export | [368199844](https://www.kaggle.com/code/toktarkhan/soup-sql-rescore-evidence-export?scriptVersionId=368199844) |
| Training and controls | [368195915](https://www.kaggle.com/code/toktarkhan/soup-take-home-training-and-controls?scriptVersionId=368195915) |
| Earlier merge/verify notebook, superseded by the amended stages | [368201856](https://www.kaggle.com/code/toktarkhan/soup-take-home-merge-quantize-and-verify?scriptVersionId=368201856) |
| Stage 1 · amendment and split | [368212358](https://www.kaggle.com/code/toktarkhan/soup-patch-stage-1-amendment-and-split?scriptVersionId=368212358) |
| Stage 2a · real merges | [368212862](https://www.kaggle.com/code/toktarkhan/soup-patch-stage-2a-real-merges?scriptVersionId=368212862) |
| Stage 2b · control merges | [368212888](https://www.kaggle.com/code/toktarkhan/soup-patch-stage-2b-control-merges?scriptVersionId=368212888) |
| Stage 3 · NF4 export | [368217419](https://www.kaggle.com/code/toktarkhan/soup-patch-stage-3-nf4-export?scriptVersionId=368217419) · [First attempt](https://www.kaggle.com/code/toktarkhan/soup-patch-stage-3-nf4-export?scriptVersionId=368215926) |
| Stage 4 · core verification | [368224256](https://www.kaggle.com/code/toktarkhan/soup-patch-stage-4-core-verification?scriptVersionId=368224256) · [First attempt](https://www.kaggle.com/code/toktarkhan/soup-patch-stage-4-core-verification?scriptVersionId=368221722) |
| Stage 5 · decoding probes | [368231043](https://www.kaggle.com/code/toktarkhan/soup-patch-stage-5-decoding-probes?scriptVersionId=368231043) |
| Stage 6 · Soup ship | [368234412](https://www.kaggle.com/code/toktarkhan/soup-patch-stage-6-soup-ship?scriptVersionId=368234412) |
| Stage 7 · diagnostics | [368244280](https://www.kaggle.com/code/toktarkhan/soup-patch-stage-7-diagnostics?scriptVersionId=368244280) |
| P12 · type validation | [368253002](https://www.kaggle.com/code/toktarkhan/soup-patch-p12-type-validation?scriptVersionId=368253002) |
| Stage 8 · remaining probes | [368251275](https://www.kaggle.com/code/toktarkhan/soup-patch-stage-8-remaining-probes?scriptVersionId=368251275) |
| P11 · synthetic execution | [368264249](https://www.kaggle.com/code/toktarkhan/soup-patch-p11-sql-execution?scriptVersionId=368264249) |
| Stage 8 · quant-check retry | [368273101](https://www.kaggle.com/code/toktarkhan/soup-patch-stage-8-quant-check-retry?scriptVersionId=368273101) |
| POST-HOC S5 · FP16 and INT8 | [368356653](https://www.kaggle.com/code/toktarkhan/soup-post-hoc-s5-fp16-and-int8?scriptVersionId=368356653) |
| POST-HOC consistency · NF4 repeat | [368367783](https://www.kaggle.com/code/toktarkhan/soup-post-hoc-nf4-consistency?scriptVersionId=368367783) |

## Model and data

The base was Qwen2.5-1.5B-Instruct at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, published under Apache-2.0. The 3B checkpoint was published under the Qwen Research License. [Pinned 1.5B release](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306) · [3B model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

The SQL data came from b-mc2’s `sql-create-context` dataset at revision `9d80a6a118b838d9defc3798d659a54a2ac2ff37`. The project deduplicated and split rows, added a chat prompt and category, and constructed shuffled and intentionally corrupted control fixtures. [Dataset and attribution](https://huggingface.co/datasets/b-mc2/sql-create-context/tree/9d80a6a118b838d9defc3798d659a54a2ac2ff37) · [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)
