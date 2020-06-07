import hydra
import torch

from omegaconf import DictConfig
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss

from dataset import get_dataloaders
from networks import get_network

from utils.logging import (
    _lf,
    log_training_loss,
    log_training_results,
    log_fields_stdout,
)


def create_supervised_trainer(cfg: DictConfig, device="cpu") -> Engine:

    # Network
    model = get_network(cfg)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    # Loss
    loss_fn = torch.nn.NLLLoss()

    def _update(engine, batch):

        ########################################################################
        # Modify the logic of your training
        ########################################################################
        model.train()
        optimizer.zero_grad()
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        # Anything you want to log must be returned in this dictionary
        update_dict = {"nll": loss.item(), "y_pred": y_pred, "y": y}
        return update_dict

    engine = Engine(_update)

    # Common metrics require base "y_pred" and "y". Lambda to reduce verbosity in metrics
    _ypred_y = lambda _: (_["y_pred"], _["y"])

    # Specify metrics. "output_transform" used to select items from "update_dict" needed by metrics
    # https://pytorch.org/ignite/metrics.html#ignite.metrics.Loss
    metrics = {
        "accuracy": Accuracy(output_transform=_ypred_y),
        "nll": Loss(loss_fn, output_transform=_ypred_y),
    }

    # Attach metrics to engine
    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


########################################################################
# Main training loop
########################################################################
@hydra.main(config_path="configs/default.yaml")
def train(cfg: DictConfig) -> None:

    # Determine device (GPU, CPU, etc.)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data Loader
    train_loader, val_loader = get_dataloaders(cfg, num_workers=cfg.data_loader_workers)

    # Training loop logic
    trainer = create_supervised_trainer(cfg, device=device)

    # Callbacks
    trainer.add_event_handler(Events.STARTED, lambda engine: print("Start training"))
    # trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results)
    # trainer.add_event_handler(
    #     Events.ITERATION_COMPLETED, lambda _: log_training_loss(_, field="nll")
    # )
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, _lf(log_fields_stdout, ["nll", "nll"])
    )

    # train()
    trainer.run(train_loader, max_epochs=cfg.mode.train.max_epochs)


if __name__ == "__main__":
    train()
