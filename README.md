# Ice Station Zebra

A pipeline for predicting sea ice.

## Setting up your environment

### Tools

You will need to install the following tools if you want to develop this project:

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

### Creating your own configuration file

Create a file in `config` that is called `<your chosen name here>.local.yaml`.
You will want this to inherit from `zebra.yaml` and then apply your own changes on top.
For example, the following config will override the `base_path` option in `zebra.yaml`:

```yaml
defaults:
  - zebra

base_path: /local/path/to/my/data
```

You can then run this with, e.g.:

```bash
uv run zebra datasets create --config-name <your local config>.yaml
```
You can also use this config to override other options in the `zebra.yaml` file, as shown below:

```yaml
defaults:
  - zebra
  - override /model: encode_unet_decode # Use this format if you want to use a different config

# Override specific model parameters
model:
  processor:
    start_out_channels: 37 # Use this format to override specific model parameters in the named configs

base_path: /local/path/to/my/data
```
Alternatively, you can apply overrides to specific options at the command line like this:

```bash
uv run zebra datasets create ++base_path=/local/path/to/my/data
```

### Running on Baskerville

As `uv` cannot easily be installed on Baskerville, you should install the `zebra` package directly into a virtual environment that you have set up.

```bash
source /path/to/venv/activate.sh
pip install -e .
```

This means that later commands like `uv run X ...` should simply be `X ...` instead.

## Running Zebra commands

### Create

You will need a [CDS account](https://cds.climate.copernicus.eu/how-to-api) to download data with `anemoi`.

Run `uv run zebra datasets create` to download all datasets locally.

### Inspect

Run `uv run zebra datasets inspect` to inspect all datasets available locally.

### Train

Run `uv run zebra train` to train using the datasets specified in the config.

:information_source: This will save checkpoints to `${BASE_DIR}/training/wandb/run-${DATE}$-${RANDOM_STRING}/checkpoints/${CHECKPOINT_NAME}$.ckpt`.

### Evaluate

Run `uv run zebra evaluate --checkpoint PATH_TO_A_CHECKPOINT` to evaluate using a checkpoint from a training run.
