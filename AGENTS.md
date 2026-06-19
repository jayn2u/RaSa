- Use `uv run python` to execute Python code.
## Dataset location

Lab datasets are stored at one of:

- `/mnt/data/lab_datasets`
- `/data/jayn2u/lab_datasets`

These paths refer to the same storage. Use whichever exists on the current machine.

Per-dataset directories (e.g. `CUHK-PEDES`, `ICFG-PEDES`, `RSTPReid`) live directly under the chosen root. Run `data_process.py` with `--dataset_root_dir` pointing to the dataset directory (e.g. `/data/jayn2u/lab_datasets/CUHK-PEDES`), then update paths in `configs/PS_*.yaml` to the processed output.
