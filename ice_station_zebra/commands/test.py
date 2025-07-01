from omegaconf import DictConfig, OmegaConf


def test(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg["test"]))
