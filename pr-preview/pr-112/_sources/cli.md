# CLI Commands

The Ice Station Zebra CLI provides commands for dataset management, model training, and evaluation.

## Main Entry Point

The main CLI entry point is accessed through:

```bash
uv run zebra --help
```

This provides access to all available commands organized into subcommands.

## Available Commands

### Dataset Management

#### Create Dataset
```bash
uv run zebra datasets create [CONFIG_NAME] [OVERRIDES...]
```

Creates a new dataset from configuration files.

**Parameters:**
- `CONFIG_NAME`: Name of the configuration file (optional, defaults to "zebra")
- `OVERRIDES`: Space-separated Hydra config overrides

**Example:**
```bash
uv run zebra datasets create era5-0d5-south-2019-12-24h-v1
```

#### Inspect Dataset
```bash
uv run zebra datasets inspect [CONFIG_NAME] [OVERRIDES...]
```

Inspects an existing dataset to show its structure and contents.

**Parameters:**
- `CONFIG_NAME`: Name of the configuration file (optional, defaults to "zebra")
- `OVERRIDES`: Space-separated Hydra config overrides

**Example:**
```bash
uv run zebra datasets inspect era5-0d5-south-2019-12-24h-v1
```

### Model Training

#### Train Model
```bash
uv run zebra train [CONFIG_NAME] [OVERRIDES...]
```

Trains a model using the specified configuration.

**Parameters:**
- `CONFIG_NAME`: Name of the configuration file (optional, defaults to "zebra")
- `OVERRIDES`: Space-separated Hydra config overrides

**Example:**
```bash
uv run zebra train encode_ddpm_decode
```

### Model Evaluation

#### Evaluate Model
```bash
uv run zebra evaluate [CONFIG_NAME] [OVERRIDES...]
```

Evaluates a trained model on test data.

**Parameters:**
- `CONFIG_NAME`: Name of the configuration file (optional, defaults to "zebra")
- `OVERRIDES`: Space-separated Hydra config overrides

**Example:**
```bash
uv run zebra evaluate default
```

## Configuration Overrides

All commands support Hydra configuration overrides using the syntax:

```bash
uv run zebra [COMMAND] [CONFIG_NAME] key=value key.subkey=value
```

**Examples:**
```bash
# Override model parameters
uv run zebra train encode_ddpm_decode model.n_forecast_steps=7

# Override data paths
uv run zebra datasets create era5-0d5-south-2019-12-24h-v1 data.path=/path/to/data

# Multiple overrides
uv run zebra train encode_ddpm_decode model.n_forecast_steps=7 trainer.max_epochs=100
```

## Getting Help

For detailed help on any command:

```bash
uv run zebra --help
uv run zebra datasets --help
uv run zebra train --help
uv run zebra evaluate --help
```

## Configuration Files

Configuration files are located in `ice_station_zebra/config/` and define:

- **Dataset configurations**: Data sources, preprocessing, and storage
- **Model configurations**: Architecture, training parameters, and optimization
- **Training configurations**: Trainer settings, callbacks, and logging
- **Evaluation configurations**: Metrics, visualization, and output formats

See the [Configuration Guide](configuration.md) for more details on creating and customizing configuration files.
