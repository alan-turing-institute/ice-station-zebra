# Ice Station Zebra

A pipeline for predicting sea ice.

## Setting up your environment

### Creating your own configuration file

Create a file in `config` that is called `<your chosen name here>.local.yaml`.
You will want this to inherit from `zebra.yaml` and then apply your own changes on top.
For example, the following config will override the `base_path` option in `zebra.yaml`:

```yaml
defaults:
  - zebra

base_path: /local/path/to/my/data
```

Alternatively, you can apply overrides at the command line like this:

```bash
uv run zebra datasets create ++base_path=/local/path/to/my/data
```

### Running on Baskerville

As `uv` cannot easily be installed on Baskerville, you should install the `zebra` package directly into a virtual environment that you have set up.

```sh
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
