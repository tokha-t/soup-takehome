# AI tool usage

I used AI for planning, implementation, execution, and review. My role was to orchestrate the process, review the proposed approach and results, and set the standard of evidence required to call the model meaningfully trained and ready to ship. The coding agent wrote and ran the implementation; I directed the investigation and reviewed its evidence and conclusions.

## Where AI was used

| Tool | Role |
|---|---|
| Claude (chat) | Interpreting the brief and Soup documentation; dataset selection; experiment specifications and revisions; memory estimates; draft verification and release criteria; post-hoc investigation; first report draft |
| A second AI reviewer | Challenging the experiment design and reviewing the report brief |
| Codex coding agent | Writing and running Kaggle code for data preparation, training, merging, quantization, verification, failure probes, Soup checks, and packaging |
| Codex final review | Reviewing the code, saved outputs, and report against the brief; replaying saved-result calculations; identifying remaining limitations; polishing the submission |

## My role and decisions

I coordinated the planning, implementation, and review stages and used the reviewers' findings to direct follow-up work. I set the checking standard around measurable task improvement, preservation of the learned change through export, and an absolute requirement for usable SQL. A successful run or falling loss was not enough for the release decision.

I accepted additional checks proposed during review: weak-adapter controls, minimum effect sizes, overlap at the individual `CREATE TABLE` level, matching comparison dtypes, batched-versus-single decoding parity, and export-memory estimates. The release criteria included at least 90% retention with an 80% lower confidence bound, and at least 95% valid SQL under the shipped settings.

I kept the proposed “three times quantization noise” comparison as a diagnostic rather than a release gate. I corrected planning that still assumed a live discussion after Soup had cancelled it. I reviewed the distinction between the built-in Soup verdict and the project's stricter release criteria; the final decision remains DON'T SHIP.

## Changes and unreliable suggestions

- Claude predicted that repetition penalty 1.1 would hurt performance. In the isolated greedy comparison, increasing it from 1.0 to 1.1 instead raised lenient match by 4.69 percentage points. Comparing sampling with greedy decoding would not isolate that effect.
- After the first post-hoc diagnosis, Claude attributed the failure entirely to the checker. Further measurements showed that the NF4 model still failed the SQL-validity requirement after the checker was repaired.
- The initial split allowed individual `CREATE TABLE` statements to appear in both training and evaluation. Eight overlapping rows were removed before the main artifact verification, leaving 192. This correction occurred after training and its periodic held-out evaluations.
- The 191-row headline table is separately labeled POST-HOC: it also excludes row 2961, whose gold query names a column missing from the schema. The original results remain archived.
- The amendment was written after training and before the main merge and verification results. Its rules are described as amended, not as fully result-blind.
- The report separates the original pre-training memory prediction from amended phase budgets and calculations using observed batch shapes.

The agent made implementation decisions during execution, including using PEFT after Soup rejected the training data format, repairing incompatible tokenizer metadata, implementing the SQL compatibility rules, and switching to single-prompt decoding when parity checks failed. These actions, failures, and limitations are documented in the [agent execution record](ai_usage_agent.md), [evidence notes](EVIDENCE_NOTES.md), and raw logs.

## Scope of my review

My review focused on the experimental design, checking standards, interpretation of results, and release decision. Code implementation, experiment execution, numerical replay, and automated verification were performed with AI assistance. The submission preserves the evidence supporting the conclusions, including failed attempts and the incomplete raw training-console capture.
