# Reviewing and reproducing the submission

The official decision remains **DON'T SHIP**. A rerun is a new experiment; it does not replace the saved results, preregistration or POST-HOC labels. See [EVIDENCE_NOTES.md](EVIDENCE_NOTES.md) for the measurement limits and [README.md](README.md#public-kaggle-notebooks) for all pinned notebook versions.

## 1. Review the saved evidence on a CPU

From the repository root, with Python 3.11 or newer:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 audit_saved_results.py
```

This standard-library-only command reads the saved files and prints its findings. It checks both registration hashes, recomputes the token-weighted answer NLL from the 192 per-row records in `results/verify_real.json`, and applies the stated POST-HOC compatibility rules to the saved generations. It neither loads a model nor writes evidence.

The local check with SQLite 3.50.4 reproduced:

| Quantity | Result |
|---|---:|
| Official base / merged FP16 answer NLL | 0.362918 / 0.078971 |
| Official base NF4 / merged NF4 answer NLL | 0.389137 / 0.108850 |
| Gold SQL accepted under POST-HOC compatibility rules | 191 / 192 |
| Valid shipped-setting SQL, seeds 42 / 43 / 44 | 164 / 166 / 169 of 191 |
| Combined valid SQL | 499 / 573 = 87.1% |
| Generations that stopped | 569 / 573 |

The audit prints the local SQLite version and any differences in recomputed validity flags. SQLite behavior is part of the check. Replaying saved outputs does not establish model identity, reproduce training, or independently validate whether the POST-HOC rule changes were appropriate.

## 2. Prepare a separate GPU working copy

All stages resolve output paths relative to their source tree. Run GPU work in a disposable copy, with its own `results`, `logs`, `runs` and `artifacts`. For example, from the parent directory of a checkout named `soup-takehome`:

```sh
cp -R soup-takehome soup-replay
cd soup-replay
mv results reference_results
mv logs reference_logs
mkdir results logs
cp reference_results/data_stats.json results/
cp reference_results/s5_posthoc.json results/
```

The second copied JSON is only needed by the POST-HOC follow-ups. Keep the frozen `data/` files, both registration documents and hashes, and all source modules. The data statistics are required: the verifier reads the registered generation cap from `results/data_stats.json`.

The recorded GPU environment was one visible Tesla T4, Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0 and driver 580.178.04; see [the recorded environment](results/environment.json). Use a Linux CUDA environment suitable for bitsandbytes and the T4. Kaggle's current image may differ from the recorded one, so record the actual environment of any new run. The code deliberately restricts visibility to GPU 0 and rejects an unexpected device in the training environment check.

For direct PEFT training or standalone verification, create an isolated environment on a GPU runtime that already has a working CUDA PyTorch installation:

```sh
python -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install transformers==4.57.1 peft==0.18.0 accelerate==1.11.0 bitsandbytes==0.48.2 sqlglot==28.0.0
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled SOUP_TELEMETRY=0 USE_TF=0
```

The amended Kaggle orchestration scripts instead create their own environments through `uv`, inheriting the notebook's PyTorch installation. They pin the PEFT packages above and a separate Soup environment with `soup-cli[train]==0.77.2`, `bitsandbytes==0.48.2` and `torchao==0.17.0`. These are the recorded application-package pins, not a complete lockfile of the hosted image.

### Fresh time budgets

The original scripts included the submission's absolute deadlines. `replay.py` supplies fresh relative deadlines to the training and amended-stage helpers before they are imported:

```sh
python replay.py --minutes 120 --optional-minutes 90 patch_stage_diagnostics.py main
```

Both budgets start at launch; optional work stops earlier when requested. The launcher passes `SOUP_REPLAY_DEADLINE_EPOCH` and `SOUP_REPLAY_OPTIONAL_EPOCH` to subprocesses. Existing per-command limits still apply. It gives the stage 15 seconds after its overall deadline to let its own command timers terminate work, then stops the stage if needed. `--gpu-monitor` adds the existing nvidia-smi sampler for direct training or verification; stage scripts that already start their own sampler do not need that flag. A literal `@deadline` argument is replaced with the new ISO timestamp for the POST-HOC entrypoints.

Without an override, the historical deadlines are unchanged. The only changes to the original orchestration are environment-aware deadline initialization in `train_stage.py` and `patch_common.py`, plus a clearer expired-deadline message. The exact pre-polish code is preserved at [initial commit f358fa5](https://github.com/tokha-t/soup-takehome/tree/f358fa5c233aefda296e14768c39400bf9a4ca38). The launcher does not alter data, models, gates, seeds or verdict logic.

## 3. Recover checkpoints and run the standalone verifier

Weights are not committed to Git. Open the **pinned notebook version**, then recover its output files, or attach that version's output to a new Kaggle notebook. Preserve complete checkpoint directories, including configuration and tokenizer files. Do not select an unpinned latest version or substitute a newly trained adapter when auditing the archived model.

| Checkpoint needed | Output provider | Output directory |
|---|---|---|
| Real LoRA adapter | Training, version 368195915 | `runs/real/adapter/` |
| Merged FP16 | Stage 2a, version 368212862 | `artifacts/merged_fp16/` |
| Base-equivalent NF4 from the zero-update placebo | Stage 3, version 368217419 | `artifacts/base_4bit/` |
| Portable shipped NF4 | Stage 4, version 368224256 | `artifacts/merged_4bit_portable/` |
| Optional FP16 controls and scaled adapters | Stage 2b, version 368212888 | `artifacts/placebo_fp16/`, `artifacts/shuffled_fp16/`, `artifacts/weak_001_fp16/`, `artifacts/weak_01_fp16/` |
| Optional shuffled NF4 control | Stage 3, version 368217419 | `artifacts/shuffled_4bit/` |

Compare recovered files with [adapter hashes](results/training_adapter_hashes.json), [FP16 hashes](results/artifact_hashes_2a.json), [control hashes](results/artifact_hashes_2b.json), [NF4 hashes](results/artifact_hashes_3.json) and [portable-artifact hashes](results/artifact_hashes_4.json). These hash lists identify the artifacts; their presence in Git alone does not supply the weights. The portable NF4 directory restores tokenizer files from the pinned base while preserving the original NF4 tensor bytes; see [packaging evidence](results/production_packaging.json).

For example, copy the four required directories into `checkpoints/real-adapter`, `checkpoints/merged_fp16`, `checkpoints/base_4bit` and `checkpoints/merged_4bit_portable`, then run from the GPU working copy:

```sh
python replay.py --minutes 120 --gpu-monitor verify.py \
  --base Qwen/Qwen2.5-1.5B-Instruct \
  --adapter checkpoints/real-adapter \
  --merged checkpoints/merged_fp16 \
  --base4 checkpoints/base_4bit \
  --merged4 checkpoints/merged_4bit_portable \
  --eval data/eval_amended.jsonl \
  --out results/replay_verify_real.json
```

The verifier pins the base to `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Add `--placebo-merged PATH`, `--shuffled-merged PATH`, `--shuffled4 PATH` and repeated `--extra NAME=PATH` arguments to reproduce the control comparisons. The amended Stage 4 command in `patch_stage_core.py` is the complete reference invocation. Read the output JSON's metrics and `verdict`; a successful process exit alone is not a passing model gate. This verifier does not by itself run every S1–S8 gate or all probes.

## 4. Reproduce the staged Kaggle workflow

The pinned notebooks are the execution record, including their input attachments and environment setup. For a new staged run, use the same stage order and attach each completed upstream output explicitly. Run the updated source from `/kaggle/working`, not directly inside the read-only `/kaggle/input` mount. A fresh source/output directory per stage keeps new evidence separate from copied reference results.

The scripts look for exact suffixes under `/kaggle/input` rather than fixed dataset slugs. Keep paths such as `runs/real/adapter/adapter_model.safetensors` intact. Provide only one copy of each large checkpoint; ambiguous matches stop the run. Identical small files may be deduplicated by hash, but differing copies must be resolved explicitly. Stage 1 is stricter: it expects exactly one mounted `prereg.md`, from the training output.

For replay from the archived adapters, begin with the pinned training output. Stage 1 audits the historical amendment and also copies the historical Soup failure log, so it needs that archived training output even if you separately repeat PEFT training. Later adapter-consuming stages can use the new PEFT output when evaluating a new run; keep new adapters and their training records together, and do not describe the copied historical failure log as a newly executed Soup attempt.

| Order | Entry point, after `python replay.py --minutes 120` | Required upstream evidence |
|---|---|---|
| Training | See frozen-input replay below | Frozen training data, config and original registration |
| 1: amendment | `patch_stage_one.py` | Training output, including original data, registration and Soup failure log; amendment files in the working source |
| 2a and 2b: merges | `patch_stage_merge.py 2a` and `patch_stage_merge.py 2b`, in separate stages | Stage 1 amendment; all three training adapters |
| 3: NF4 export | `patch_stage_quant.py` | Stage 1 amended data/statistics; training adapters; 2a/2b FP16 outputs for the fallback path |
| 4: verification | `patch_stage_core.py` | Training adapters, Stage 1 data and amendment, all 2a/2b/3 artifacts and Stage 3 result/hash evidence |
| 5: decoding | `patch_stage_decode.py` | Stage 1 data/statistics and amendment; Stage 4 portable checkpoint and verification result |
| 6: Soup ship | `patch_stage_ship.py` | Training real/placebo adapters; Stage 1 ChatML training/task files, statistics and amendment; Stage 4 portable checkpoint |
| 7: diagnostics | `patch_stage_diagnostics.py main` | Training records/adapters; Stage 1 data; merge, quantization and verification outputs from 2a/2b/3/4 |
| 8: remaining probes | `patch_stage_optional.py main` | Stage 1 data; training adapters; artifacts and results from the preceding stages, as attached in pinned notebook 368251275 |
| P11 retry | `patch_stage_execution.py` | Inputs attached in pinned notebook 368264249 |
| Quant-check retry | `patch_stage_quant_check.py main` | Inputs attached in pinned notebook 368273101 |

Use the exact Inputs lists of the pinned versions when replaying optional probes: they recover additional prior outputs by suffix. The separate P12 type-validation notebook, version 368253002, records its own CPU commands. The bake-off and rescore notebooks precede training and document the original data-selection decision; rerunning the frozen experiment does not require making that decision again.

### Frozen-input training replay

The executed training path fell back to PEFT after Soup rejected the source format. To rerun those three training jobs against the frozen inputs, use the prepared GPU working copy and pinned PEFT environment:

```sh
python -c 'from instruments import setup_logging, record_environment; setup_logging(); record_environment()'
python replay.py --minutes 120 --gpu-monitor configs/train_peft.py real > logs/train_real.log 2>&1
python replay.py --minutes 120 --gpu-monitor configs/train_peft.py placebo > logs/train_placebo.log 2>&1
python replay.py --minutes 120 --gpu-monitor configs/train_peft.py shuffled > logs/train_shuffled.log 2>&1
```

Keep the frozen data and `results/data_stats.json`; each job checks the original registration, saves its adapter under `runs/`, and writes structured training records under `results/`. The commands also retain console output in `logs/`. This replays the measured PEFT path, not the earlier unsuccessful Soup attempt. New training may produce different bytes on a changed runtime. Use its own hashes and downstream outputs consistently, and label them as a new run.

`train_stage.py` is the historical initial orchestration: it also rebuilds data and creates a preregistration from an external `pipeline_spec.md`, which is not part of this repository, and it refuses to overwrite an existing registration. It is not a one-command entry point for the frozen submission. `eval_stage.py` likewise records an earlier, superseded merge/evaluation attempt and uses obsolete verifier arguments; use the amended stages above or the standalone command in section 3.

### POST-HOC follow-ups on the archived model

Attach the same pinned inputs as the two POST-HOC notebooks, keep the copied `results/s5_posthoc.json`, and run:

```sh
python replay.py --minutes 120 s5_followup.py remote @deadline
python replay.py --minutes 120 consistency_nf4.py remote @deadline
```

The S5 follow-up needs the amended evaluation data, Stage 5 decoding evidence, the FP16 merge and portable tokenizer. The consistency follow-up needs the original real adapter, amended evaluation data and Stage 4 verification/portable artifact. Its hardcoded adapter hash deliberately checks the original archived adapter; a newly trained adapter is a different experiment and will fail that guard. These follow-ups remain POST-HOC and do not retroactively change the official S5 miss.

## Checks performed on this reproduction support

`python3 -m unittest discover -s tests -v` checks historical fallback deadlines, invalid budgets, refreshed required/optional/training timeouts, expired-budget behavior and propagation to a real lightweight subprocess. The read-only CPU audit was also run. The documentation and deadline update have **not** been validated by a new full GPU replay; the original pinned notebooks remain the evidence for the reported model runs.
