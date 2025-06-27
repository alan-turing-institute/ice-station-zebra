from pathlib import Path

from omegaconf import DictConfig, OmegaConf


class TrainerConfigAdaptor(DictConfig):
    def __init__(self, config: DictConfig):

        print(OmegaConf.to_yaml(config, resolve=True))

        training_path = Path(config["data_path"]) / "training"

        datasets = {
            "concat": [
                f"{config["data_path"]}/anemoi/{dataset_name}.zarr"
                for dataset_name in config["datasets"].keys()
            ]
        }
        print(datasets)

        frequencies = set(
            config["datasets"][dataset_name]["dates"]["frequency"]
            for dataset_name in config["datasets"].keys()
        )
        if len(frequencies) != 1:
            raise ValueError(
                "Found {len(frequencies)} dataset frequencies when exactly one was expected"
            )

        super().__init__(
            content={
                "data": {
                    "forcing": [],  # features that are not part of the forecast state but are used as forcing to generate the forecast state
                    "format": "zarr",
                    "frequency": list(frequencies)[0],
                    "timestep": list(frequencies)[0],
                },
                "dataloader": {
                    "pin_memory": True,
                    "read_group_size": "${hardware.num_gpus_per_model}",
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
                    "checkpoint": config["train"]["checkpointing"],
                    "debug": {
                        "anomaly_detection": False  # trace back NaNs at the cost of slowing down training
                    },
                    "enable_checkpointing": config["train"]["enable_checkpointing"],
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
                                "every_n_epochs": "epoch_trigger-epoch_{epoch:03d}-step_{step:06d}",
                                "every_n_train_steps": "step_trigger-epoch_{epoch:03d}-step_{step:06d}",
                                "every_n_minutes": "time_trigger-epoch_{epoch:03d}-step_{step:06d}",
                            },
                        },
                        "paths": {
                            "checkpoints": training_path / "checkpoints",
                            "graph": training_path / "graphs",
                            "plots": training_path / "plots",
                        },
                    },
                    **config["train"]["hardware"],
                ),
                "graph": "",
                "model": "",
                "training": {
                    "deterministic": False,
                    "fork_run_id": None,
                    "load_weights_only": False,  # only load model weights, do not restore optimiser states etc.
                    "lr": config["train"]["learning"],
                    "max_epochs": None,
                    "max_steps": 150000,
                    "multistep_input": 1,  # how many previous timesteps to use when training model
                    "rollout": {
                        "epoch_increment": 0,  # increase rollout every n epochs
                        "start": 0,
                    },
                    "run_id": None,
                },
                "config_validation": False,
            }
        )
