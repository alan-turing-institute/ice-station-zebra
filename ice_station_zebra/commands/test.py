import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../config", config_name="zebra")
def test(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg["test"]))