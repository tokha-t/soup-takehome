# Agent log

- Used seed 42 and a single visible T4 with the pinned Qwen revision.
- Used nearest-rank percentiles across the deduplicated pool and reported the 600-row sample separately.
- Set the generation cap at twice the pool p95 for answer tokens.
- Supervised the assistant end-of-turn token during training. Reported held-out loss on answer content tokens and measured stopping separately.
- Used non-reentrant gradient checkpointing in the PEFT probe to keep the same batch size of 8 on one T4.
- Marked N1 when at least 90% of answers were between 5 and 100 tokens.
- Used p95 rendered length, then p50, to resolve a remaining sequence-length tie.
- Reported normalized prompt overlap and reused context overlap independently.
- Reported the sampled device memory during the training window.
- Used the signed-in Kaggle website after the connector returned Unauthenticated. The T4 selector showed 30 GPU hours remaining before launch.

## Issues and deviations

- The first Kaggle attempt stopped during environment setup because Python's ensurepip exited 1. The retry used Kaggle's bundled uv to create the environment.
- On the second attempt, Soup 0.77.2 completed data doctor, then soup train --yes exited 1 before training. Its hardware-fit gate estimated 28.7 GB peak VRAM (31.6 GB with margin) against 15.6 GB available. The full command output is in kaggle_attempt2.log. The remaining probes use Transformers + PEFT.
- The initial PEFT pass ran out of memory on B and C. The final pass enabled non-reentrant gradient checkpointing for all three candidates with the same batch size of 8, model precision and training hyperparameters.

## Dataset and scoring decisions

- Used Google Mobile Actions rows with exactly one tool call from the publisher's training split. Put the seven shared tool signatures in the fixed system prompt and kept each row's time and weekday in the user prompt.
- Used MASSIVE English training rows with at least one annotated slot. Supplied the published intent and scored verbatim slot spans as arrays in occurrence order.
- Kept the first row for each normalized duplicate prompt. The SQL pool contained one duplicate prompt with conflicting references. Mobile Actions contained a near duplicate with a different timestamp. Neither changed the zero exact train-to-held-out prompt overlap.
- Marked N2 when at least half the deduplicated examples copied a value from the prompt. The pool metrics and manual-review row IDs are in the JSON and the CPU screening log.
- Left the SQL harder slice untested because its base score was 0%, below the spec's trigger of above 70%.

## Run and file verification

- The completed run is [Kaggle version 3](https://www.kaggle.com/code/toktarkhan/soup-dataset-bake-off?scriptVersionId=368173550). It used Transformers + PEFT after the earlier Soup hardware-fit rejection. The portable notebook starts with the Soup attempt when rerun.
- Chrome blocked the output download. Two private CPU notebooks exported the exact saved files for the second and final runs. No GPU training ran in those export notebooks.
- Matched the five final run files to their Kaggle SHA-256 hashes before adding this section. Checked 100 paired correctness values per candidate, all flip totals and gains, and all 657 final device samples. Preserved the first two attempt logs and the second attempt's 344 device samples separately.
