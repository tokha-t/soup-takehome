# Bake-off failure evidence

- The virtual environment setup failed because ensurepip exited 1. [First attempt log](kaggle_attempt1.log)
- Soup predicted 28.7 GB peak, or 31.6 GB including its margin, against 15.6 GB available. [Second attempt log](kaggle_attempt2.log)
- The same attempt recorded separate B and C PEFT CUDA out-of-memory tracebacks. The final bake-off reran the three probes with gradient checkpointing and batch size 8. [Second attempt log](kaggle_attempt2.log) · [Final bake-off log](bakeoff.log)
