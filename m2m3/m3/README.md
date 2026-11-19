# Milestone 3 ? Evaluation, Analysis and Demo

## Scope

M3 evaluates five model configurations over seeds 42, 43 and 44
on the locked custom `filtered_original` CrisisMMD protocol.

- Train: 6,001 pairs
- Development: 986 pairs
- Held-out test: 951 pairs

Primary implementation: `../experiment.py`.
Exploratory CLIP and residual-gated scripts are separate from
the reported primary 15-run experiment.

## Environment

From the repository root:

```bash
python -m pip install -r m2m3/requirements.txt
python -m pytest -q m2m3/tests
python m2m3/experiment.py check --repo .
```

## Primary experiment reproduction

Requires the licensed source data, pretrained encoder weights,
sufficient compute resources, and fresh output directories.

```bash
python m2m3/download_mirror.py --repo . --out data/mirror
python m2m3/experiment.py extract --repo . --mirror data/mirror --out features/primary --device auto
python m2m3/run_all.py --repo . --features features/primary --runs runs/primary --milestone all
python m2m3/experiment.py finalize --repo . --features features/primary --runs runs/primary --out results/m3_reproduced --bootstrap 500
```

The final held-out test evaluation must only be performed after
development-based selection. Do not tune models using test results.
Existing nonempty output directories are intentionally protected
against overwriting.

## Local M3 analysis and demo

Analysis scripts can use the environment variable
`CO3133_M3_DATA_ROOT`. The default is the `M3_Local_Input`
directory next to the repository.

The local workspace contains feature caches, run outputs,
checkpoints and, where necessary, licensed raw media.
These inputs are not bundled with the published source.

For example, in PowerShell:

```powershell
$env:CO3133_M3_DATA_ROOT = (Resolve-Path "..\M3_Local_Input").Path
python m2m3/m3/run_interactive_demo.py
```

The demo requires the original trained checkpoints and encoder
dependencies. It is not a publicly deployed service.

## Public evidence

- `../../results/m3/`: aggregate development, test, ablation,
  bootstrap and efficiency results.
- `../../docs/m3/`: public review and milestone documentation.
- `../../artifacts/m2/`: nine existing M2 head checkpoints.

All fifteen primary run logs and development predictions are available across `artifacts/m2/` and `artifacts/m3/`. The six new M3 head checkpoints are in `artifacts/m3/`; the nine matching M2 checkpoints remain in `artifacts/m2/`. Per-seed test metrics and their SHA256 checksums are published. Run `python scripts/verify_m3_evidence.py` from the repository root. Full independent GPU reproduction is not yet established.

Do not publish raw crisis media, private galleries, credentials,
or sensitive runtime files.
