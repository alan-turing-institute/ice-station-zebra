import inspect
import itertools
from collections.abc import Callable
from typing import Annotated, ParamSpec, TypeVar

from hydra import compose, initialize
from omegaconf import DictConfig
from typer import Argument, Option

Param = ParamSpec("Param")
RetType = TypeVar("RetType")


def hydra_adaptor(function) -> Callable[Param, RetType]:
    """Replace a function that takes a Hydra config with one that takes string arguments

    Args:
        function: Callable(*args, config: DictConfig, **kwargs)

    Returns:
        Callable(*args, config_name: str, **kwargs, overrides: list[str])
    """

    def wrapper(
        overrides: Annotated[
            list[str] | None,
            Argument(
                help="Apply space-separated Hydra config overrides (https://hydra.cc/docs/advanced/override_grammar/basic/)"
            ),
        ] = None,
        config_name: Annotated[
            str | None,
            Option(help="Specify the name of a file to load from the config directory"),
        ] = "zebra",
        *args: Param.args,
        **kwargs: Param.kwargs,
    ) -> RetType:
        with initialize(config_path="../config", version_base=None):
            config = compose(config_name=config_name, overrides=overrides)
        return function(*args, config=config, **kwargs)

    # Remove the DictConfig parameter from the function signature
    fn_signature = inspect.signature(function, eval_str=True)
    function_params = (
        param
        for param in fn_signature.parameters.values()
        if param.annotation != DictConfig
    )

    # Take only the overrides and config_name names from the function signature
    additional_params = (
        param
        for param in inspect.signature(wrapper, eval_str=True).parameters.values()
        if param.name in ("overrides", "config_name")
    )

    # Since the additional parameters are keyword arguments we can simply append them
    combined_parameters = list(itertools.chain(function_params, additional_params))
    wrapper.__signature__ = fn_signature.replace(parameters=combined_parameters)
    wrapper.__name__ = function.__name__
    return wrapper
