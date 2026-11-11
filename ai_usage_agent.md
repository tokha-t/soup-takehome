# Agent and AI usage

- Generated the notebook staging code, dataset builder, pre-registration and memory calculator, fallback training, merge and NF4 orchestration, verifiers, probes, and result assembly.
- Used the signed-in Kaggle website after the connector returned Unauthenticated. The rescore and main work ran as separate private notebook sessions.
- Regenerated the original 500-row LoRA probe because the bake-off saved correctness flags without generated text. Main training began only after hashing the main-run registration.
- Trained with peft because Soup rejected the data format (`pre_tokenized row must have 'input_ids'.`). Fixed by converting the same 1,500 rows to strictly validated ChatML after training. All post-training gates are Soup's. The SQL verifier and release thresholds are additional project measurements.
- The original pre-registration was left intact. The amendment was written after training and before post-training results. Removed eight affected rows from the main evaluation without retraining.
- Used PEFT for the fp16 and bf16 saves after Soup merge failed, and for the fp32 reference because Soup has no fp32 save format. Retried NF4 export with a compatible TorchAO; all three selected exports use the same Soup command.
- Replaced incompatible tokenizer metadata with the pinned Qwen files while retaining the exact Soup NF4 weights. Stopped the first core run when generation defaults overrode the requested controls; used the completed rerun. Two shuffled-model parity failures triggered single-row decoding.
- Soup refused the lr=0 benchmark before training. The initial P12 fixture had empty message lists that Soup counted as valid; the follow-up used 750 type-invalid rows to exercise the threshold.
- Per-eval peak resets and new training instrumentation could not be added to completed training runs. The post-training phases and separate real benchmark report their own peaks.
- Reran P11 on CPU after a local reproduction showed the read-only authorizer could block fixture commit. The corrected probe commits the fixture before evaluating either query.
- Result decisions use the amended gates. Unmeasured criteria count as misses at the deadline.

## Failed or unavailable steps

- bench_placebo recorded exit 1; retain its raw log and distinguish a runtime/input rejection from a measured negative result.
- quant_check_real recorded exit 3; retain its raw log and distinguish a runtime/input rejection from a measured negative result.
- quant_check_wrong recorded exit 3; retain its raw log and distinguish a runtime/input rejection from a measured negative result.
- probe_P11_execution recorded exit 1; retain its raw log and distinguish a runtime/input rejection from a measured negative result.
- ship_noise_floor_3 recorded exit -15; retain its raw log and distinguish a runtime/input rejection from a measured negative result.
- quant_check_real recorded exit -15; retain its raw log and distinguish a runtime/input rejection from a measured negative result.

## POST-HOC analysis and report

- Reused saved generations for the SQLite compatibility diagnosis, error breakdown, retry estimate and exact paired tests. Ran the requested FP16 sampling, INT8 greedy probe and NF4 repeat in separately recorded Kaggle notebooks.
- Resolved the report’s placeholders from saved evidence, computed the 191-row table from archived per-row records, and verified the PDF’s page geometry, font size and page count.
- Prepared the public GitHub evidence package and set the committed Kaggle notebooks to public. The two preregistration files and the official summary were carried over byte for byte.
