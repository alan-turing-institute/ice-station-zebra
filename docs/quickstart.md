# Quickstart

Get your environment ready to run Ice Station Zebra.

## Prerequisites

- Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) to manage environments and execute Zebra commands.

## Configure your run

Create a configuration file in `config` named `<your chosen name>.local.yaml`. Inherit from `base.yaml` and override values you need:

```yaml
defaults:
  - base

base_path: /local/path/to/my/data
```

Run any Zebra command with your config:

```bash
uv run zebra <command> --config-name <your local config>.yaml
```

You can also override specific options within the config:

```yaml
defaults:
  - base
  - override /model: encode_unet_decode

model:
  processor:
    start_out_channels: 37

base_path: /local/path/to/my/data
```

Or override values at the command line:

```bash
uv run zebra <command> ++base_path=/local/path/to/my/data
```

Note: `persistence.yaml` already overrides the options in `base.yaml` required to run the `Persistence` model.

## Running on Baskerville

`uv` cannot easily be installed on Baskerville. Instead, install the `zebra` package directly into a virtual environment:

```bash
source /path/to/venv/activate.sh
pip install -e .
```

Subsequent commands become `zebra <command> ...` instead of `uv run zebra <command> ...`.

## Next steps

- See detailed [CLI commands](cli.md) for dataset, training, and evaluation workflows.
- Explore the [tutorial notebooks](tutorials/index.md) to learn hands-on usage.

