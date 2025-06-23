from hydra import compose, initialize
from typer import Argument


def hydra_override(function):
    def wrapper(
        overrides: list[str] | None = Argument(None), config_name: str = "zebra"
    ):
        with initialize(config_path="config", version_base=None):
            config = compose(config_name=config_name, overrides=overrides)
        return function(config)

    return wrapper