"""Assemble measured facts, threshold decisions and unresolved checks."""
from datetime import datetime
from copy import deepcopy
import json
import sys
from zoneinfo import ZoneInfo

from instruments import ROOT, atomic_json
from predict_memory import amended_predictions

RESULTS = ROOT / "results"


def interval_positive(metric):
    return bool(metric and metric.get("ci95") and metric["ci95"][0] > 0)


def memory_phases(facts):
    phases = []
    for name, values in facts.items():
        if not name.startswith("memory_phases_") or not isinstance(values, list):
            continue
        for phase in values:
            row = {**phase, "source": f"results/{name}.json"}
            for key in tuple(row):
                value = row[key]
                if key.endswith("_bytes") and any(term in key for term in ("memory", "peak", "ram")):
                    row[key.removesuffix("_bytes") + "_gib"] = value / 2**30 if value is not None else None
                    del row[key]
                if key.startswith("nvidia") and key.endswith("_mib") and value is not None:
                    row[key.removesuffix("_mib") + "_gib"] = value / 1024
            phases.append(row)
    return phases


def report_measurements(facts):
    """Keep normalized measurements in the summary and preserve raw tool files."""
    values = {name: value for name, value in facts.items() if not name.startswith("memory_phases_")}
    if "memory_prediction" in values:
        values["memory_prediction"] = {"status": "superseded by memory_predictions_amended",
                                       "historical_source": "results/memory_prediction.json"}
    if "soup_bench_real" in values:
        bench = values["soup_bench_real"] = deepcopy(values["soup_bench_real"])
        bench["memory"] = {name.removesuffix("_bytes") + "_gib": count / 2**30
                           for name, count in bench["memory"].items()}
        bench["source"] = "results/soup_bench_real.json"
    if "memory_grid" in values:
        grid = values["memory_grid"] = deepcopy(values["memory_grid"])
        for point in grid["points"]:
            point.pop("soup_profile", None)
    return values


def read_soup_json_log(relative):
    path = ROOT / relative
    candidates = (path.with_stem(path.stem + "_attempt2"), path)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        body = [line.partition(" INFO ")[2] for line in candidate.read_text(encoding="utf-8").splitlines()]
        for index, line in enumerate(body):
            if line.lstrip().startswith("{"):
                try:
                    obj, _ = json.JSONDecoder().raw_decode("\n".join(body[index:]).lstrip())
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    return {"source": str(candidate.relative_to(ROOT)), "result": obj}
    return None


def decision(evidence, *, final=False):
    real = evidence.get("verify_real", {})
    b = real.get("bootstrap", {})
    flips = real.get("paired_flips", {})
    d = real.get("distances", {})
    p4 = evidence.get("probe_P4", {})
    sample = p4.get("sampling_pooled", {}).get("all", {})
    gold = p4.get("gold_valid_sql", real.get("gold_sqlite_calibration", {}).get("rate"))
    stand = evidence.get("standalone", {})
    suite = evidence.get("ship_b_evidence", {}).get("benchmarks", {})
    suite_counts = evidence.get("soup_eval_source", {}).get("suite_counts", {})
    suite_source = "ship_b_evidence"
    if not suite and (live := evidence.get("ship_b_live_log_excerpt", {}).get("benchmarks_displayed")):
        suite = {name: {"base": row["base"], "tuned": row["tuned"]} for name, row in live.items()}
        suite_counts = {name: row["items"] for name, row in live.items()}
        suite_source = "ship_b_live_log_excerpt"
    suite_drops = {name: {**pair, "drop": pair["base"]-pair["tuned"], "items": suite_counts.get(name)}
                   for name, pair in suite.items()}
    controls = {name: evidence.get("verify_"+name, {}).get("verdict", {}).get("training")
                for name in ("real", "placebo", "shuffled", "weak_001", "weak_01")}
    expected_controls = {"real": {"TRAINED"}, "placebo": {"NOT_TRAINED"}, "weak_001": {"NOT_TRAINED"},
                         "shuffled": {"NOT_TRAINED", "TRAINED_NOT_ON_TASK"}}
    control_results = [controls[name] in allowed if controls[name] else None for name, allowed in expected_controls.items()]
    s8 = False if False in control_results else True if all(x is True for x in control_results) else None
    r = b.get("retention", {})
    stopping = sample["stop"] >= .99 if sample else None
    sql_validity = sample["valid_sql"] >= .95 if sample and gold is not None and gold >= .99 else None
    calibrated_sql = False if False in (stopping, sql_validity) else True if stopping and sql_validity else None
    sql_fix = ("For a fresh frozen evaluation, repair the schema and gold-query errors listed in verify_real.json, verify at least 99% gold validity with SQLite EXPLAIN QUERY PLAN, and measure the shipped decoder against the 95% threshold."
               if gold is not None and gold < .99 else
               "Retest a decoder with a reliable end-of-turn stop and SQL-valid output, targeting 99% stop and 95% valid SQL.")
    suite_complete = bool(suite_drops and suite_counts and suite_counts.keys() <= suite_drops.keys())
    s = {
        "S1": (interval_positive(b["training"]) and real["verdict"]["MIN_LOSS"] if "training" in b else None,
               "Improve the real adapter until the lower 95% bound for the relative answer-loss reduction reaches 10%."),
        "S2": (interval_positive(b["content_4bit"]) if "content_4bit" in b else None,
               "Correct the training targets or quantization so the real NF4 model beats the identically quantized shuffled control in paired answer loss."),
        "S3": (bool(r.get("estimate") is not None and r["estimate"] >= .90 and r["ci95"] and r["ci95"][0] >= .80
                    and flips["gain"] >= .10 and flips["exact_mcnemar_p"] < .01) if r and flips else None,
               "Preserve at least 90% of the loss effect (80% lower bound), then recheck the paired 10-point lenient-match gain and McNemar p value."),
        "S4": (d["fingerprint_share"] >= .95 if "fingerprint_share" in d else None,
               "Use the verified real merge as quantizer input and remeasure the fingerprint against the two reference distributions."),
        "S5": (calibrated_sql, sql_fix),
        "S6": (all(row["drop"] <= .05 + 1e-12 for row in suite_drops.values()) if suite_complete else None,
               "Recover every failing general suite to within 0.05 of its base score and rerun the same Soup gate."),
        "S7": (stand.get("standalone") is True and p4["shipped_match_cost"] <= .02 + 1e-12 if stand and "shipped_match_cost" in p4 else None,
               "Ship the verified standalone tokenizer and template with the measured greedy decoding settings, then repeat the three-seed comparison."),
        "S8": (s8, "Investigate the labels outside their registered expectations and repeat verification before release."),
    }
    s6_attempted = "stage_6" in evidence
    missed = any(value is False or (value is None and (final or key == "S6" and s6_attempted)) for key, (value, _) in s.items())
    return {
        "verdict": "SHIP" if all(value is True for value, _ in s.values()) else "DON'T SHIP" if missed else "PENDING",
        "gates": {key: {"pass": value, "status": "PASS" if value is True else "FAIL" if value is False else "UNMEASURED",
                        "counts_as_miss": value is False or (value is None and (final or key == "S6" and s6_attempted)),
                        "action_if_missed": None if value is True else fix} for key, (value, fix) in s.items()},
        "general_suite_drops": suite_drops, "general_suite_source": suite_source if suite else None, "controls": controls,
        "S5_components": {
            "stop_pass": stopping, "sql_validity_pass": sql_validity, "gold_validity": gold,
            "gold_rows": p4.get("gold_rows", real.get("eval_rows")), "gold_calibration_minimum": .99,
            "sampled_outputs": sample.get("rows"), "stop_rate": sample.get("stop"),
            "sql_validity_uncalibrated": sample.get("valid_sql"), "sql_validity_minimum": .95,
        },
    }


def summarize(*, final=False):
    facts = {path.stem: json.loads(path.read_bytes()) for path in sorted(RESULTS.glob("*.json")) if path.name != "summary.json"}
    phases = memory_phases(facts)
    if phases:
        atomic_json(RESULTS / "memory_phases.json", phases)
        facts["memory_phases"] = phases
    diagnostic_logs = facts.get("diagnostic_commands", {})
    adapter_checks = {}
    for name, item in diagnostic_logs.items():
        if name.startswith("adapters_check_") and (parsed := read_soup_json_log(item["log"])):
            adapter_checks[name.removeprefix("adapters_check_")] = parsed
    if adapter_checks:
        atomic_json(RESULTS / "soup_adapter_checks.json", adapter_checks)
        facts["soup_adapter_checks"] = adapter_checks
    if audit := diagnostic_logs.get("adapters_audit_real"):
        if result := read_soup_json_log(audit["log"]):
            atomic_json(RESULTS / "soup_adapter_audit.json", result)
            facts["soup_adapter_audit"] = result
    verdict = decision(facts, final=final)
    hashes = {key: value for name, manifest in facts.items() if name.startswith("artifact_hashes_")
              for key, value in manifest.items()}
    hashes.update({f"adapter_{key}": value for key, value in facts.get("training_adapter_hashes", {}).items()})
    if "production_artifact" in facts:
        hashes["merged_4bit_soup_raw"] = hashes["merged_4bit"]
        hashes["merged_4bit"] = hashes["merged_4bit_portable"]
    predictions = amended_predictions()
    optional_evidence = {
        label: f"results/{label}.json" if label in facts else None
        for label in ("memory_grid", "quant_check_real", "quant_check_wrong", "ship_d", "ship_weak_001",
                      "weights_A3_nf4", "probe_P2", "replayed_training_padding", "probe_P9", "probe_P10",
                      "probe_P11", "ship_noise_floor_3")
    }
    optional_evidence["merged_4bit_from_fp32"] = (
        "results/artifact_hashes_8q.json" if "merged_4bit_from_fp32" in hashes else None
    )
    atomic_json(RESULTS / "summary.json", {**verdict, "artifact_hashes": hashes,
                "memory_predictions_amended": predictions, "optional_evidence": optional_evidence,
                "measurements": report_measurements(facts)})
    lines = ["# Measured findings", ""]
    r = facts.get("rescore", {})
    if r.get("base"):
        lines += [
            f"- The SQL rescore measured {r['base']['strict']:.1%} strict and {r['base']['lenient']:.1%} lenient base match. [rescore.json](results/rescore.json)",
            f"- The recreated LoRA probe reached {r['probe']['lenient']:.1%} lenient match. [rescore.json](results/rescore.json)",
            f"- The rescore gold validity was {r['gold_calibration']['rate']:.1%}. [rescore.json](results/rescore.json)",
        ]
    data = facts.get("data_stats", {})
    if data:
        lines += [
            f"- Deduplication removed {data['dedup_dropped']} rows. [data_stats.json](results/data_stats.json)",
            f"- The original whole-context splits have {data['train']['rows']} train and {data['eval']['rows']} eval rows. [data_stats.json](results/data_stats.json)",
            f"- The maximum rendered length was {max(data['train']['full_tokens']['max'],data['eval']['full_tokens']['max'])}, yielding a configured limit of {data['max_length']}. [data_stats.json](results/data_stats.json)",
        ]
    split = facts.get("amendment_split", {})
    if split:
        lines.append(f"- The audit found {split['shared_statement_count']} shared CREATE TABLE statements and excluded {len(split['excluded_eval_rows'])} rows. The remaining 192 eval rows share zero such statements with training. [amendment_split.json](results/amendment_split.json)")
    lines.append("- Soup rejected `pre_tokenized` rows without `input_ids`. The same 1,500 rows validated as ChatML after PEFT training. [Raw training error](logs/failed/soup_format/raw.log) · [Stage 1](results/stage_1.json)")
    for name in ("1", "2a", "2b", "3", "4", "5", "6", "7", "7p12", "8", "8p11", "8q"):
        key = f"stage_{name}"
        if stage := facts.get(key):
            lines.append(f"- Kaggle stage {name} finished at {stage['finished']} ({stage['status']}). [{key}.json](results/{key}.json)")
    if (ROOT / "logs/failed/core_generation_defaults/raw.log").is_file():
        lines.append("- The first verification attempt was cancelled after the log showed inherited sampling defaults. Its scores are excluded. [Raw log](logs/failed/core_generation_defaults/raw.log)")
    if pack := facts.get("production_artifact"):
        lines.append(f"- The packaged NF4 model loaded standalone and retained the identical Soup weight files. [Packaging record](results/production_artifact.json) · [Standalone evidence]({pack['standalone_proof']})")
    real = facts.get("verify_real", {})
    if real:
        for name, metric in real["bootstrap"].items():
            if name != "loss":
                estimate = metric["estimate"]
                lines.append(f"- {name.replace('_',' ').capitalize()} measured {estimate if estimate is not None else 'undefined'}, with 95% interval {metric['ci95']}. [verify_real.json](results/verify_real.json)")
        for name, metrics in real["generations"].items():
            row = metrics["all"]
            lines.append(f"- {name} scored {row['lenient']:.1%} lenient match, {row['valid_sql']:.1%} valid SQL and {row['stop']:.1%} stopping. [verify_real.json](results/verify_real.json)")
        lines.extend([
            f"- The fp16 merge changed {real['merge_fp16']['changed_count']} tensors. [verify_real.json](results/verify_real.json)",
            f"- Casting the published base from bf16 to fp16 changed {real['merge_fp16']['base_bf16_to_fp16_changed_elements']:,} elements of {real['merge_fp16']['base_bf16_elements_compared']:,} compared. [verify_real.json](results/verify_real.json)",
            f"- The production NF4 fingerprint was closer to the tuned merge on {real['distances']['fingerprint_share']:.1%} of rows. [verify_real.json](results/verify_real.json)",
            f"- Gold validity on the main eval set was {real['gold_sqlite_calibration']['rate']:.1%}. [verify_real.json](results/verify_real.json)",
            f"- Repeating base greedy decoding changed {real['instrument_noise_outputs_different']} of {real['eval_rows']} outputs. [verify_real.json](results/verify_real.json)",
        ])
        flips = real["paired_flips"]
        lines.append(f"- NF4 versus fp16 base flipped {flips['wrong_to_right']} rows to correct and {flips['right_to_wrong']} to wrong, for a {flips['gain']*100:.2f}-point gain (exact McNemar p={flips['exact_mcnemar_p']:.4g}). [verify_real.json](results/verify_real.json)")
        for label, entries in real["padding_parity"].items():
            for entry in entries:
                if not entry["equal"]:
                    lines.append(f"- {label} greedy decoding differed on parity rows {entry['mismatch_ids']}; the verifier scored it one row at a time. [verify_real.json](results/verify_real.json)")
        for name, metric in real["bootstrap"]["loss"].items():
            lines.append(f"- {name} answer NLL was {metric['estimate']:.5f} per token. [verify_real.json](results/verify_real.json)")
    for label, verdict_label in verdict["controls"].items():
        if verdict_label:
            key = f"verify_{label}"
            row = facts[key]["distances"]
            extra = ""
            if label in ("real", "weak_001", "weak_01") and row.get("quantization_noise_ratio") is not None:
                flag = " (below 3; descriptive flag)" if row.get("ratio_below_3") else ""
                extra = f" Its training-to-quantization KL ratio was {row['quantization_noise_ratio']:.3f}{flag}."
            lines.append(f"- {label} was classified {verdict_label}.{extra} [{key}.json](results/{key}.json)")
    for run in ("real", "placebo", "shuffled"):
        key = "train_"+run
        if run_metrics := facts.get(key):
            lines.append(f"- The {run} run logged {len(run_metrics['step_metrics'])} optimizer steps and {run_metrics['skipped_steps']} skips. Throughput excluding warmup and eval was {run_metrics['non_pad_tokens_per_second_excluding_warmup_and_eval']:.0f} non-pad tokens/s, with {run_metrics['padding_share']:.1%} padding. [{key}.json](results/{key}.json)")
    training_phases = {f"{run}_{stage}" for run in ("real", "placebo", "shuffled")
                       for stage in ("model_load", "training", "adapter_save")}
    for phase in phases:
        if (phase["name"] in training_phases | {"real_eval_core", "probe_decoding_4bit", "production_standalone_load_generate"}
                or phase["name"].startswith(("merge_", "quant_"))):
            device = phase.get("device") or "CUDA"
            rss = phase.get("peak_process_rss_gib", phase.get("peak_rss_gib"))
            allocated = phase.get("torch_peak_allocated_gib")
            gpu = f" and GPU allocated peak {allocated:.3f} GiB" if allocated is not None else ""
            if rss is not None:
                finished = datetime.fromtimestamp(phase["end_epoch"], ZoneInfo("Asia/Almaty")).isoformat(timespec="seconds")
                state = phase.get("state", "recorded")
                lines.append(f"- {phase['name']} ({device}, {state}) ended at {finished} with peak RSS {rss:.3f} GiB{gpu}. [Phase log]({phase['source']})")
    for key, metrics in verdict["general_suite_drops"].items():
        source = verdict["general_suite_source"]
        lines.append(f"- Soup's {key} suite changed from {metrics['base']:.3f} to {metrics['tuned']:.3f} on {metrics['items']} items. [{source}.json](results/{source}.json)")
    if p4 := facts.get("probe_P4"):
        row = p4["sampling_pooled"]["all"]
        lines.append(f"- The shipped decoder produced {row['rows']} outputs across three seeds. {row['stop']:.1%} stopped before the cap, {row['valid_sql']:.1%} parsed, and sampled lenient match minus greedy was {-p4['shipped_match_cost']*100:+.2f} points. [probe_P4.json](results/probe_P4.json)")
        if ablation := p4.get("greedy_1_1"):
            delta = (ablation["all"]["lenient"] - p4["greedy_1_0"]["all"]["lenient"]) * 100
            lines.append(f"- At greedy decoding, changing the repetition penalty from 1.0 to 1.1 changed lenient match by {delta:+.2f} points. [probe_P4.json](results/probe_P4.json)")
    if p3 := facts.get("probe_P3"):
        lines.append(f"- Removing the system prompt changed lenient match from {p3['with_system']['all']['lenient']:.1%} to {p3['without_system']['all']['lenient']:.1%}. [probe_P3.json](results/probe_P3.json)")
    if p1 := facts.get("probe_P1"):
        lines.append(f"- At length {p1['max_length']}, {p1['either_cut']}/{p1['rows']} rows lost answer or stop supervision ({p1['assessment']}). [probe_P1.json](results/probe_P1.json)")
    if p12 := facts.get("probe_P12"):
        lines.append(f"- Replacing half the rows with empty message lists returned exit {p12['exit_codes']['half_default']} by default and {p12['exit_codes']['half_strict']} with a 100% valid-row threshold; Soup counted empty message lists as valid. [probe_P12.json](results/probe_P12.json)")
    if p12 := facts.get("probe_P12_type_corruption"):
        lines.append(f"- The type-invalid P12 fixture returned exit {p12['exit_codes']['half_default']} by default and {p12['exit_codes']['half_strict']} with a 100% valid-row threshold. [probe_P12_type_corruption.json](results/probe_P12_type_corruption.json)")
    if bench := facts.get("soup_bench_real"):
        allocated = bench["memory"]["max_memory_allocated_bytes"] / 2**30
        reserved = bench["memory"]["max_memory_reserved_bytes"] / 2**30
        lines.append(f"- Soup's real training benchmark measured {bench['throughput']['useful_tokens_per_second']:.1f} supervised tokens/s over {bench['timing']['counted_steps']} timed steps, peaking at {allocated:.3f} GiB allocated and {reserved:.3f} GiB reserved. [soup_bench_real.json](results/soup_bench_real.json)")
    if lr0 := diagnostic_logs.get("bench_placebo"):
        lines.append(f"- Soup rejected the zero-learning-rate benchmark at config validation (exit {lr0['exit_code']}); its parameter-change check did not run. [Raw log]({lr0['log']})")
    if p5 := facts.get("probe_P5"):
        for dtype, row in p5["dtypes"].items():
            rounded = row["rounded_to_zero_share"]
            lines.append(f"- The {dtype} merge rounded away {rounded:.6%} of nonzero expected edits and scored {row['answer_loss']['estimate']:.5f} answer NLL with {row['forward_dtype']} compute. [probe_P5.json](results/probe_P5.json)")
    if wrong := facts.get("probe_P6"):
        lines.append(f"- The wrong NF4 checkpoint's fingerprint was closer to the tuned merge on {wrong['wrong']['fingerprint_share']:.1%} of rows. [probe_P6.json](results/probe_P6.json)")
    for label, value in facts.get("ship_checkpoints", {}).items():
        lines.append(f"- Soup ship ({label}) finished at {value['finished']} with exit {value['exit_code']}. [Soup checkpoint record](results/ship_checkpoints.json)")
    for label in ("b", "a", "c", "shuffled", "d", "weak_001", "noise_floor_3"):
        key = f"ship_{label}"
        if result := facts.get(key):
            task = result["task_win"]
            detail = ("The adapter_noop rule fired before model loading." if result.get("failed_rule") == "adapter_noop"
                      else f"Exact task match was {task['base']:.1%} for the base and {task['tuned']:.1%} for the candidate.")
            lines.append(f"- Soup ship ({label}) returned {result['decision']}. {detail} [{key}.json](results/{key}.json)")
    if b := facts.get("ship_checkpoints", {}).get("b"):
        binding = (b.get("provenance") or {}).get("binding") or {}
        if binding.get("task_eval_sha256"):
            lines.append("- Soup's NF4 evidence binds the task file, config, source data at verdict time, suite revision and policy. The checkpoint record separately records the SHA-256 of the tested NF4 weight file. [Soup checkpoint record](results/ship_checkpoints.json)")
    if a3 := facts.get("weights_A3_nf4"):
        lines.append(f"- The real NF4 export changed {a3['total_code_flip_rate']:.2%} of codes versus the identically exported base. [weights_A3_nf4.json](results/weights_A3_nf4.json)")
    for target in ("real", "wrong"):
        key = f"quant_check_{target}"
        if result := facts.get(key):
            for row in result.get("rows", []):
                lines.append(f"- Soup quant-check for the {target} artifact scored {row['before']:.1%} before and {row['after']:.1%} after, giving {row['verdict']}. [{key}.json](results/{key}.json)")
    if grid := facts.get("memory_grid"):
        for row in grid.get("points", []):
            measured = row.get("measured_allocated_gib")
            detail = f"{measured:.3f} GiB allocated" if measured is not None else row.get("error", "unmeasured")
            soup = row.get("soup_estimate_gib")
            estimate = f"; Soup estimated {soup:.3f} GiB" if soup is not None else ""
            lines.append(f"- The memory grid at checkpointing {row['checkpointing']}, batch {row['batch']}, length {row['length']} recorded {detail}{estimate}. [memory_grid.json](results/memory_grid.json)")
        for mode, fit in grid.get("fits", {}).items():
            lines.append(f"- The {mode} memory fit had fixed cost {fit['fixed_gib']:.3f} GiB and slope {fit['slope_mib_per_token']:.3f} MiB per padded token. [memory_grid.json](results/memory_grid.json)")
    if p2 := facts.get("probe_P2"):
        lines.append(f"- With pad equal to EOS, token-ID masking would remove the supervised stop from {p2['eos_masked_by_token_id']}/{p2['rows']} rows; attention-mask masking would remove it from {p2['eos_masked_by_attention_mask']}. [probe_P2.json](results/probe_P2.json)")
        eos_check = next((x for x in p2["soup_gate"].get("checks", []) if x["name"] == "eos_in_labels"), None)
        if eos_check:
            lines.append(f"- Soup data doctor marked its EOS-label check {eos_check['verdict']} on the pad-equals-EOS tokenizer. [probe_P2_soup.json](results/probe_P2_soup.json)")
    if p9 := facts.get("probe_P9"):
        lines.append(f"- P9 found {p9['nonfinite_rows']}/{p9['eval_rows']} nonfinite fp16 forwards. Across {p9['paired_prompts']} paired prompts, maximum absolute logit difference was {p9['max_abs_logit_difference']:.4f} and fp32-to-fp16 KL was {p9['kl_fp32_to_fp16']:.6g}. [probe_P9.json](results/probe_P9.json)")
    if p10 := facts.get("probe_P10"):
        lines.append(f"- Lenient match was {p10['schema_disjoint']['all']['lenient']:.1%} on the clean split and {p10['deliberately_leaky']['all']['lenient']:.1%} on the intentionally overlapping cohort; category proportions also differ. [probe_P10.json](results/probe_P10.json)")
    if p11 := facts.get("probe_P11"):
        for name, row in p11["models"].items():
            if row["usable"]:
                lines.append(f"- Synthetic-fixture execution match for {name} was {row['execution_match']:.1%} on {row['usable']} usable rows, versus {row['lenient_on_same_rows']:.1%} lenient match; {row['both_empty']} cases had both result sets empty. [probe_P11.json](results/probe_P11.json)")
    if replay := facts.get("replayed_training_padding"):
        for name, row in replay["runs"].items():
            lines.append(f"- Replaying the {name} sampler and collator without a model forward matched its historical token counts: {row['matches_original_logged_counts']}. [replayed_training_padding.json](results/replayed_training_padding.json)")
    for group, stage in (("optional_commands", "8"), ("quant_check_commands", "8q")):
        for name, row in facts.get(group, {}).items():
            code = row.get("exit_code")
            outcome = "unmeasured at cutoff" if code is None else f"exit {code}"
            links = (row.get("log"), f"results/{name}.json", f"logs/stages/{stage}/{name}.log", f"results/{group}.json")
            link = next(candidate for candidate in links if candidate and (ROOT / candidate).is_file())
            lines.append(f"- Optional step {name} finished at {row.get('finished', 'unrecorded time')} with {outcome}. [Evidence]({link})")
    if (facts.get("optional_commands", {}).get("ship_noise_floor_3", {}).get("exit_code") == -15
            and (ROOT / "logs/stages/8/ship_noise_floor_3_attempt2.log").is_file()):
        lines.append("- The three-repeat noise-floor gate timed out twice without producing a verdict. [First attempt](logs/stages/8/ship_noise_floor_3.log) · [Second attempt](logs/stages/8/ship_noise_floor_3_attempt2.log)")
    for name, item in facts.get("diagnostic_commands", {}).items():
        lines.append(f"- {name} finished at {item['finished']} with exit {item['exit_code']}. [Raw log]({item['log']})")
    for name, item in adapter_checks.items():
        if label := item["result"].get("verdict"):
            lines.append(f"- Soup adapters check labelled {name} {label}. [Raw log]({item['source']})")
    if audit := facts.get("soup_adapter_audit"):
        a = audit["result"]
        lines.append(f"- Soup's real-adapter audit checked {a['checked_count']} settings, reported {a['diverged_count']} divergences, and marked {a['unknown_count']} unknown. [soup_adapter_audit.json](results/soup_adapter_audit.json)")
    if final:
        for label, source in optional_evidence.items():
            if source is None:
                lines.append(f"- The optional {label} check did not produce a measurement. [summary.json](results/summary.json)")
    for gate, state in verdict["gates"].items():
        label = {True: "passed", False: "failed", None: "unmeasured (counts as a miss)" if state["counts_as_miss"] else "unmeasured"}[state["pass"]]
        lines.append(f"- {gate} was {label}. [summary.json](results/summary.json)")
    lines.append(f"- The registered decision was {verdict['verdict']}. [summary.json](results/summary.json)")
    (ROOT / "findings.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    status = facts.get("eval_stage", {})
    issues = [f"- {name} recorded {row['exit_code']} after {row['attempts']} attempt(s)."
              for name, row in status.items() if row["exit_code"] not in (0,2)]
    for group in ("diagnostic_commands", "optional_commands", "quant_check_commands"):
        for name, row in facts.get(group, {}).items():
            if row.get("exit_code") not in (0, 2):
                issues.append(f"- {name} recorded exit {row.get('exit_code')}; retain its raw log and distinguish a runtime/input rejection from a measured negative result.")
    notes = [
        "# Agent and AI usage", "",
        "- Generated the notebook staging code, dataset builder, pre-registration and memory calculator, fallback training, merge and NF4 orchestration, verifiers, probes, and result assembly.",
        "- Used the signed-in Kaggle website after the connector returned Unauthenticated. The rescore and main work ran as separate private notebook sessions.",
        "- Regenerated the original 500-row LoRA probe because the bake-off saved correctness flags without generated text. Main training began only after hashing the main-run registration.",
        "- Trained with peft because Soup rejected the data format (`pre_tokenized row must have 'input_ids'.`). Fixed by converting the same 1,500 rows to strictly validated ChatML after training. All post-training gates are Soup's. The SQL verifier and release thresholds are additional project measurements.",
        "- The original pre-registration was left intact. The amendment was written after training and before post-training results. Removed eight affected rows from the main evaluation without retraining.",
        "- Used PEFT for the fp16 and bf16 saves after Soup merge failed, and for the fp32 reference because Soup has no fp32 save format. Retried NF4 export with a compatible TorchAO; all three selected exports use the same Soup command.",
        "- Replaced incompatible tokenizer metadata with the pinned Qwen files while retaining the exact Soup NF4 weights. Stopped the first core run when generation defaults overrode the requested controls; used the completed rerun. Two shuffled-model parity failures triggered single-row decoding.",
        "- Soup refused the lr=0 benchmark before training. The initial P12 fixture had empty message lists that Soup counted as valid; the follow-up used 750 type-invalid rows to exercise the threshold.",
        "- Per-eval peak resets and new training instrumentation could not be added to completed training runs. The post-training phases and separate real benchmark report their own peaks.",
        *(["- Reran P11 on CPU after a local reproduction showed the read-only authorizer could block fixture commit. The corrected probe commits the fixture before evaluating either query."]
          if facts.get("stage_8p11", {}).get("status") == "complete" else []),
        "- Result decisions use the amended gates. Unmeasured criteria count as misses at the deadline.",
        "", "## Failed or unavailable steps", "", *(issues or ["- None in the recorded evaluation stage."]),
    ]
    (ROOT / "ai_usage_agent.md").write_text("\n".join(notes)+"\n", encoding="utf-8")


if __name__ == "__main__":
    now = datetime.now(ZoneInfo("Asia/Almaty"))
    summarize(final="--final" in sys.argv or now >= datetime(2026, 11, 11, 10, tzinfo=now.tzinfo))
