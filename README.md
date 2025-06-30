# Ice Station Zebra

A pipeline for predicting sea ice.

## Creating your own configuration file

Create a file in `config` that is called `<your chosen name here>.local.yaml`.
You will want this to inherit from `zebra.yaml` and then apply your own changes on top.
For example, the following config will override the `data_path` option in `zebra.yaml`:

```yaml
defaults:
  - zebra

data_path: /local/path/to/my/data
```

Alternatively, you can apply overrides at the command line like this:

```bash
uv run zebra datasets create ++data_path=/local/path/to/my/data
```

## Create

You will need a [CDS account](https://cds.climate.copernicus.eu/how-to-api) to download data with `anemoi`.

Run `uv run zebra datasets create` to download all datasets locally.
