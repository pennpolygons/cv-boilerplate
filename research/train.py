
import logging
import hydra
import torch
import torch.nn as nn

from omegaconf import DictConfig
from ignite.utils import setup_logger
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss

from dataset import get_dataloaders
from networks import get_network

from utils.image_utils import inverse_mnist_preprocess

from utils.engine_logging import (
    _lf,
    _lf_val,
    log_engine_output,
    log_engine_metrics,
    LOG_OP,
)


def add_file_pointers_dict(engine: Engine) -> None:
    engine.state.fp = {}


def create_training_loop(
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
        update_dict = {"nll": loss.item(), "y_pred": y_pred, "y": y, "im": (inverse_mnist_preprocess(x)[0] * 255).type(torch.uint8).squeeze()}
        return update_dict

    engine = Engine(_update)

    # (Optional) Specify training metrics. "output_transform" used to select items from "update_dict" needed by metrics
    # Collecting metrics over training set is not recommended
    # https://pytorch.org/ignite/metrics.html#ignite.metrics.Loss

    return engine


def create_evaluation_loop(
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

    # Specify evaluation metrics. "output_transform" used to select items from "infer_dict" needed by metrics
    # https://pytorch.org/ignite/metrics.html#ignite.metrics.
    get_ypred_and_y = lambda infer_dict: (infer_dict["y_pred"], infer_dict["y"])
    metrics = {
        "accuracy": Accuracy(output_transform=get_ypred_and_y),
        "nll": Loss(loss_fn, output_transform=get_ypred_and_y),
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
    trainer = create_training_loop(model, cfg, device=device)

    # Evaluation loop logic
    evaluator = create_evaluation_loop(model, cfg, device=device)

    trainer.logger = setup_logger("trainer")
    evaluator.logger = setup_logger("evaluator")

    ########################################################################
    # Callbacks
    ########################################################################

    #!!!! Required. Do not change. !!!!#
    trainer.add_event_handler(Events.STARTED, add_file_pointers_dict)
    evaluator.add_event_handler(Events.STARTED, add_file_pointers_dict)

    # When batch completes, log train_engine nll output for batch.
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=50),
        _lf(
            log_engine_output,
            {
                LOG_OP.SAVE_IMAGE: ["im"],
                LOG_OP.LOG_MESSAGE: ["nll"],  # Log fields as message in logfile
                LOG_OP.SAVE_IN_DATA_FILE: ["nll"],  # Log fields as separate files
            },
        ),
    )

    # When epoch completes, run evaluator engine on val_loader, then log ["accuracy", "nll"] metrics to file and stdout.
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        _lf_val(
            log_engine_metrics,
            evaluator,
            val_loader,
            {
                # LOG_OP.PRINT: ["accuracy"],
                LOG_OP.LOG_MESSAGE: ["nll", "accuracy"],
                LOG_OP.SAVE_IN_DATA_FILE: ["accuracy"],
            },
        ),
    )

    # Execute training
    trainer.run(train_loader, max_epochs=cfg.mode.train.max_epochs)


if __name__ == "__main__":
    train()
