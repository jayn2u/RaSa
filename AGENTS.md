- Use `uv run python` to execute Python code.
## Dataset location

Lab datasets are stored at one of:

- `/mnt/data/lab_datasets`
- `/data/jayn2u/lab_datasets`

These paths refer to the same storage. Use whichever exists on the current machine.

Per-dataset directories (e.g. `CUHK-PEDES`, `ICFG-PEDES`, `RSTPReid`) live directly under the chosen root. Run `data_process.py` with `--dataset_root_dir` pointing to the dataset directory (e.g. `/data/jayn2u/lab_datasets/CUHK-PEDES`), then update paths in `configs/PS_*.yaml` to the processed output.

## Config injection and dataset naming

Lab runs must select a dataset through an explicit config path. Do not rely on silent code defaults when adding or running train/eval wrappers.

### Principles

1. **Explicit dataset selection.** Every run must inject the correct per-dataset YAML. Do not assume CUHK-PEDES or any other dataset when the config path is omitted.
2. **Fail on missing injection.** Lab evaluation entry points should require an injected config (environment variable or shell wrapper). If injection is missing or empty, raise an error instead of falling back to a generic default.
3. **Per-dataset filename suffix.** When creating YAML configs for train or eval, include the dataset slug in the filename suffix.
4. **Explicit `dataset` field.** Each evaluation YAML must set `dataset` to the slug (see table below). Do not rely on code defaults for dataset selection.
5. **YAML over CLI.** Prefer YAML configs for lab evaluation wrappers. `sugarcrepe-pedes.py` rejects CLI arguments; put all run settings in the YAML file.
6. **Supported lab datasets.** Use annotations and images under `{lab_datasets_root}/CUHK-PEDES`, `{lab_datasets_root}/ICFG-PEDES`, and `{lab_datasets_root}/RSTPReid` (see [Dataset location](#dataset-location) for root paths).

### Config layout

| Role | Prefix / pattern | Entry point | Injection |
|------|------------------|-------------|-----------|
| Training / retrieval | `PS_*.yaml` | `Retrieval.py` | `--config` (shell scripts under `shell/` pass the per-dataset file) |
| Compositional eval | `sugarcrepe_*.yaml` | `sugarcrepe-pedes.py` | `SUGARCREPE_CONFIG` |

Shared model settings: `configs/config_bert.json`.

### Per-dataset files and slugs

| Dataset directory | Config slug (`dataset:`) | Training config | Compositional eval config |
|-------------------|--------------------------|-----------------|---------------------------|
| `CUHK-PEDES` | `cuhk-pedes` | `configs/PS_cuhk_pedes.yaml` | `configs/sugarcrepe_cuhk_pedes.yaml` |
| `ICFG-PEDES` | `icfg-pedes` | `configs/PS_icfg_pedes.yaml` | `configs/sugarcrepe_icfg_pedes.yaml` |
| `RSTPReid` | `rstpreid` | `configs/PS_rstp_reid.yaml` | `configs/sugarcrepe_rstp_reid.yaml` |

`PS_*.yaml` files encode dataset-specific paths (`train_file`, `test_file`, `*_image_root`, etc.) and pair with the matching checkpoint in compositional configs (`ras_config`, `checkpoint`).

### Run examples

Training (shell wrappers set `--config` explicitly):

```bash
# shell/cuhk-train.sh, shell/icfg-train.sh, shell/rstp-train.sh
python -m torch.distributed.run ... Retrieval.py --config configs/PS_cuhk_pedes.yaml ...
```

Compositional evaluation (inject per-dataset YAML):

```bash
export SUGARCREPE_CONFIG=configs/sugarcrepe_icfg_pedes.yaml
uv run python sugarcrepe-pedes.py
```

Or run all three datasets via `shell/sugarcrepe_all.sh`, which sets `SUGARCREPE_CONFIG` for each `configs/sugarcrepe_*.yaml` file in turn.
