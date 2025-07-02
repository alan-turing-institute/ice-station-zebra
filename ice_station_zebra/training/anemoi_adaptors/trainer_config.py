from pathlib import Path

from omegaconf import DictConfig, OmegaConf



class TrainerConfigAdaptor(DictConfig):
    def __init__(self, config: DictConfig):

        # Load the resolved config into Python format
        config = OmegaConf.to_container(config, resolve=True)

        training_path = Path(config["base_path"]) / "training"
        anemoi_data_path = Path(config["base_path"]) / "data" / "anemoi"

        # Get list of datasets
        datasets = {
            "concat": [
                f"{anemoi_data_path}/{dataset_name}.zarr"
                for dataset_name in config["datasets"].keys()
            ]
        }

        # Get data frequencies
        frequencies = set(
            config["datasets"][dataset_name]["dates"]["frequency"]
            for dataset_name in config["datasets"].keys()
        )
        if len(frequencies) != 1:
            msg = f"Found {len(frequencies)} dataset frequencies when exactly one was expected"
            raise ValueError(msg)

        # Load checkpointing data
        checkpointing = config["train"]["checkpointing"]
        enable_checkpointing = checkpointing.pop("enable_checkpointing", False)

        super().__init__(
            content={
                "data": {
                    "format": "zarr",
                    "frequency": list(frequencies)[0],
                    "timestep": list(frequencies)[0],
                    **config["train"]["data"],
                },
                "dataloader": {
                    "limit_batches": {
                        "test": None,  # only run over N batches if set
                        "training": None,  # only run over N batches if set
                        "validation": None,  # only run over N batches if set
                    },
                    "pin_memory": True,
                    "read_group_size": r"${hardware.num_gpus_per_model}",
                    "test": dict(
                        {"dataset": datasets}, **config["train"]["split"]["test"]
                    ),
                    "training": dict(
                        {"dataset": datasets}, **config["train"]["split"]["training"]
                    ),
                    "validation": dict(
                        {"dataset": datasets}, **config["train"]["split"]["validation"]
                    ),
                },
                "datamodule": {
                    "_target_": "ice_station_zebra.training.anemoi_adaptors.NonGriddedDataModule",
                },
                "diagnostics": {
                    "callbacks": [],
                    "checkpoint": checkpointing,
                    "debug": {
                        "anomaly_detection": False  # trace back NaNs at the cost of slowing down training
                    },
                    "enable_checkpointing": enable_checkpointing,
                    "enable_progress_bar": True,
                    "log": config["train"]["logging"],
                    "plot": {  # diagnostic plots
                        "callbacks": [],
                    },
                    "profiler": False,  # activate the pytorch profiler for debugging
                },
                "hardware": dict(
                    {
                        "files": {
                            "graph": "",
                            "checkpoint": {
                                "every_n_epochs": r"epoch_trigger-epoch_{epoch:03d}-step_{step:06d}",
                                "every_n_train_steps": r"step_trigger-epoch_{epoch:03d}-step_{step:06d}",
                                "every_n_minutes": r"time_trigger-epoch_{epoch:03d}-step_{step:06d}",
                            },
                        },
                        "paths": {
                            "checkpoints": training_path / "checkpoints",
                            "graph": training_path / "graphs",
                            "plots": training_path / "plots",
                        },
                    },
                    **config["train"]["hardware"]["physical"],
                ),
                "graph": "",
                "model": "",
                "training": {
                    "accum_grad_batches": config["train"]["learning"][
                        "accum_grad_batches"
                    ],
                    "deterministic": False,
                    "fork_run_id": None,
                    "gradient_clip": {
                        "val": 0,  # don't clip
                        "algorithm": "norm",  # default
                    },
                    "load_weights_only": False,  # only load model weights, do not restore optimiser states etc.
                    "lr": config["train"]["learning"]["learning"],
                    "max_epochs": None,
                    "max_steps": 150000,
                    "multistep_input": 1,  # how many previous timesteps to use when training model
                    "num_sanity_val_steps": 1,  # run N batches of validation before training as a sanity check
                    "precision": config["train"]["hardware"]["precision"],
                    "rollout": {
                        "epoch_increment": 0,  # increase rollout every n epochs
                        "start": 0,
                    },
                    "run_id": None,
                    "strategy": config["train"]["hardware"]["strategy"],
                },
                "config_validation": False,
            }
        )
