# Ice Station Zebra

A pipeline for predicting sea ice.

## Setting up your environment

### Tools

You will need to install the following tools if you want to develop this project:

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

### Creating your own configuration file

Create a file in `config` that is called `<your chosen name here>.local.yaml`.
You will want this to inherit from `base.yaml` and then apply your own changes on top.
For example, the following config will override the `base_path` option in `base.yaml`:

```yaml
defaults:
  - base

base_path: /local/path/to/my/data
```

You can then run this with, e.g.:

```bash
uv run zebra <command> --config-name <your local config>.yaml
```
You can also use this config to override other options in the `base.yaml` file, as shown below:

```yaml
defaults:
  - base
  - override /model: encode_unet_decode # Use this format if you want to use a different config

# Override specific model parameters
model:
  processor:
    start_out_channels: 37 # Use this format to override specific model parameters in the named configs

base_path: /local/path/to/my/data
```

Alternatively, you can apply overrides to specific options at the command line like this:

```bash
uv run zebra <command> ++base_path=/local/path/to/my/data
```

Note that `persistence.yaml` overrides the specific options in `base.yaml` needed to run the `Persistence` model.

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

N.b. There is a slightly different process for downloading very large datasets - see below.

### Inspect

Run `uv run zebra datasets inspect` to inspect all datasets available locally.

### Train

Run `uv run zebra train` to train using the datasets specified in the config.

:information_source: This will save checkpoints to `${BASE_DIR}/training/wandb/run-${DATE}$-${RANDOM_STRING}/checkpoints/${CHECKPOINT_NAME}$.ckpt`.

### Evaluate

Run `uv run zebra evaluate --checkpoint PATH_TO_A_CHECKPOINT` to evaluate using a checkpoint from a training run.

## Adding a new model

### Background

An `ice-station-zebra` model needs to be able to run over multiple different datasets with different dimensions.
These are structured in `NTCHW` format, where:
- `N` is the batch size,
- `T` is the number of history (forecast) steps for inputs (outputs)
- `C` is the number of channels or variables
- `H` is a height dimension
- `W` is a width dimension

`N` and `T` will be the same for all inputs, but `C`, `H` and `W` might vary.

Taking as an example, a batch size (`N=2`), 3 history steps and 4 forecast steps, we will have `k` inputs of shape `(2, 3, C_k, H_k, W_k)` and one output of shape `(2, 4, C_out, H_out, W_out)`.

### Standalone models

A standalone model will need to accept a `dict[str, TensorNTCHW]` which maps dataset names to an `NTCHW` Tensor of values.
The model might want to use one or more of these for training, and will need to produce an output with shape `N, T, C_out, H_out, W_out`.

As can be seen in the example below, a separate instance of the model is likely to be needed for each output to be predicted.

![image](docs/assets/pipeline-standalone.png)

Pros:
- all input variables are available without transformation

Cons:
- hard to add new inputs
- hard to add new outputs

### Processor models

A processor model is part of a larger encode-process-decode step.
Start by defining a latent space as `(C_latent, H_latent, W_latent)` - in the example below, this has been set to `(10, 64, 64)`.
The encode-process-decode model automatically creates one encoder for each input and one decoder for each output.
The dataset-specific encoder takes the input data and converts it to shape `(N, T, C_latent, H_latent, W_latent)`.
The `k` encoded datasets can then be combined in latent space to give a single dataset of shape `(N, T, k * C_latent, H_latent, W_latent)`.

This is then passed to the processor, which must accept input of shape `(N, T, k * C_latent, H_latent, W_latent)` and produce output of the same shape.

This output is then passed to one or more output-specific decoders which take input of shape `(N, T, k * C_latent, H_latent, W_latent)` and produce output of shape `(N, T, C_out, H_out, W_out)`.

![image](docs/assets/pipeline-encode-process-decode.png)

Pros:
- easy to add new inputs
- easy to add new outputs

Cons:
- input variables have been transformed into latent space

## Jupyter notebooks

There are various demonstrator Jupyter notebooks in the `notebooks` folder.
You can run these with `uv run --group notebooks jupyter notebook`.

A good one to start with is `notebooks/demo_pipeline.ipynb` which gives a more detailed overview of the pipeline.

## Downloading large datasets
For particularly large datasets, e.g. the full ERA5 dataset, it may be necessary to download the data in parts. To do this, you need to use the following sequence of commands:

```bash
uv run zebra datasets init --config-name <your config>.yaml
```

Then load each part in turn using:

```bash
uv run zebra datasets load --config-name <your config>.yaml --parts 1/n
```

When all the parts are loading, finalise the dataset with:

```bash
uv run zebra datasets finalise --config-name <your config>.yaml
```
