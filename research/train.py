import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs/default.yaml")
def train(cfg: DictConfig) -> None:
    print(cfg.pretty())


if __name__ == "__main__":
    train()
