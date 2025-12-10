# CLI commands

All commands run via `uv run zebra ...` unless noted otherwise.

## Dataset management

- You need a [CDS account](https://cds.climate.copernicus.eu/how-to-api) to download data with `anemoi`.
- Download all datasets locally:

```bash
uv run zebra datasets create --config-name <your local config>.yaml
```

- Inspect datasets available locally:

```bash
uv run zebra datasets inspect --config-name <your local config>.yaml
```

## Training

Train using the datasets specified in your config:

```bash
uv run zebra train --config-name <your local config>.yaml
```

Checkpoints are stored under `${BASE_DIR}/training/wandb/run-${DATE}$-${RANDOM_STRING}/checkpoints/${CHECKPOINT_NAME}$.ckpt`.

## Evaluation

Evaluate a saved checkpoint:

```bash
uv run zebra evaluate --checkpoint PATH_TO_A_CHECKPOINT --config-name <your local config>.yaml
```

