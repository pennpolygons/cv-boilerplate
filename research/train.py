import hydra
import torch

import torch.nn as nn
from omegaconf import DictConfig
from ignite.utils import setup_logger
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss

from dataset import get_dataloaders
from networks import get_network

from utils.engine_logging import (
    _lf,
    _lf_val,
    log_engine_output,
    log_engine_metrics,
)


def create_supervised_trainer(
    model: nn.Module, cfg: DictConfig, device="cpu"
) -> Engine:

    # Network
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

    # (Optional) Specify training metrics. "output_transform" used to select items from "update_dict" needed by metrics
    # Collecting metrics over training set is not recommended
    # https://pytorch.org/ignite/metrics.html#ignite.metrics.Loss

    return engine


def create_supervised_evaluator(
    model: nn.Module, cfg: DictConfig, device="cpu"
) -> Engine:

    # Loss
    loss_fn = torch.nn.NLLLoss()

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

        # Anything you want to log must be returned in this dictionary
        infer_dict = {"y_pred": y_pred, "y": y}
        return infer_dict

    evaluator = Engine(_inference)

    # Common metrics require base "y_pred" and "y". Lambda to reduce verbosity in metrics
    _ypred_y = lambda _: (_["y_pred"], _["y"])

    # Specify evaluation metrics. "output_transform" used to select items from "infer_dict" needed by metrics
    # https://pytorch.org/ignite/metrics.html#ignite.metrics.Loss
    metrics = {
        "accuracy": Accuracy(output_transform=_ypred_y),
        "nll": Loss(loss_fn, output_transform=_ypred_y),
    }

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


########################################################################
# Main training loop
########################################################################
@hydra.main(config_path="configs/default.yaml")
def train(cfg: DictConfig) -> None:

    # Determine device (GPU, CPU, etc.)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    model = get_network(cfg)

    # Data Loader
    train_loader, val_loader = get_dataloaders(cfg, num_workers=cfg.data_loader_workers)

    # Training loop logic
    trainer = create_supervised_trainer(model, cfg, device=device)

    # Evaluation loop logic
    evaluator = create_supervised_evaluator(model, cfg, device=device)

    trainer.logger = setup_logger("trainer")
    evaluator.logger = setup_logger("evaluator")

    ########################################################################
    # Callbacks
    ########################################################################

    # When epoch completes, run evaluator engine on val_loader, then log ["accuracy", "nll"] metrics to file and stdout.
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        _lf_val(
            log_engine_metrics, evaluator, val_loader, ["accuracy", "nll"], stdout=True,
        ),
    )

    # When batch completes, log train_engine nll output for batch.
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=50), _lf(log_engine_output, ["nll", "nll"]),
    )

    # Execute training
    trainer.run(train_loader, max_epochs=cfg.mode.train.max_epochs)


if __name__ == "__main__":
    train()
