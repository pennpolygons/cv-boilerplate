import logging
import hydra
import torch

import numpy as np
import torch.nn as nn

from typing import Dict
from omegaconf import DictConfig
from ignite.utils import setup_logger, manual_seed
from ignite.handlers import Checkpoint, ModelCheckpoint, DiskSaver
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss

from dataset import get_dataloaders
from metrics import Mape
from networks import get_network
from utils.log_operations import LOG_OP
from utils.visdom_utils import VisPlot, VisImg
from utils.image_utils import inverse_mnist_preprocess
from utils.LogDirector import LogDirector, EngineStateAttr, LogTimeLabel


def create_classification_training_loop(
    model: nn.Module, cfg: DictConfig, name: str, device="cpu"
) -> Engine:

    # Network
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.mode.train.learning_rate, momentum=0.8
    )

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

        # Anything you want to log or use to compute something to log must be returned in this dictionary
        update_dict = {
            "nll": loss.item(),
            "nll_2": loss.item() + 0.5,
            "y_pred": y_pred,
            "x": x,
            "y": y,
        }

        return update_dict

    engine = Engine(_update)

    # Required to set up logging
    engine.logger = setup_logger(name=name)

    # (Optional) Specify training metrics. "output_transform" used to select items from "update_dict" needed by metrics
    # Collecting metrics over training set is not recommended
    # https://pytorch.org/ignite/metrics.html#ignite.metrics.Loss

    return engine


def create_classification_evaluation_loop(
    model: nn.Module, cfg: DictConfig, name: str, device="cpu"
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

    engine = Engine(_inference)

    # Required to set up logging
    engine.logger = setup_logger(name=name)

    # Specify evaluation metrics. "output_transform" used to select items from "infer_dict" needed by metrics
    # https://pytorch.org/ignite/metrics.html#ignite.metrics.
    metrics = {
        "accuracy": Accuracy(
            output_transform=lambda infer_dict: (infer_dict["y_pred"], infer_dict["y"])
        ),
        "nll": Loss(
            loss_fn,
            output_transform=lambda infer_dict: (infer_dict["y_pred"], infer_dict["y"]),
        ),
    }

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_regression_training_loop(
    model: nn.Module, cfg: DictConfig, name: str, device="cpu"
) -> Engine:

    # Network
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.mode.train.learning_rate)

    # Loss
    loss_fn = torch.nn.MSELoss()

    mape_metric = Mape(
        output_transform=lambda infer_dict: (infer_dict["y_pred"], infer_dict["y"])
    )

    def _update(engine, batch):

        model.train()

        optimizer.zero_grad()
        x, y = batch
        x, y = x.to(device), y.to(device)

        y_pred = model(x)

        loss = loss_fn(10 * y_pred, 10 * y)

        loss.backward()
        optimizer.step()

        factor = 1777.0

        mse_val = loss_fn(factor * y_pred, factor * y).item()
        y_hat = y[0].cpu().detach().numpy() * factor
        y_pred_hat = y_pred[0].cpu().detach().numpy() * factor

        # Anything you want to log must be returned in this dictionary
        update_dict = {
            "loss": mse_val,
            "y_pred": y_pred * factor,
            "ypred_first": [y_hat, y_pred_hat],
            "y": y * factor,
        }

        return update_dict

    engine = Engine(_update)

    # Required to set up logging
    engine.logger = setup_logger(name=name)

    metrics = {"mape": mape_metric}

    for name, metric in metrics.items():
        print("attaching metric:" + name)
        metric.attach(engine, name)

    return engine


def create_regression_evaluation_loop(
    model: nn.Module, cfg: DictConfig, name: str, device="cpu"
) -> Engine:

    # Loss
    loss_fn = torch.nn.MSELoss()

    mape_metric = Mape(
        output_transform=lambda infer_dict: (infer_dict["y_pred"], infer_dict["y"])
    )

    def _inference(engine, batch):

        model.eval()

        with torch.no_grad():

            x, y = batch
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            factor = 1777.0

            y = y * factor
            y_pred = y_pred * factor
            mse_val = loss_fn(y, y_pred).item()

        infer_dict = {
            "loss": mse_val,
            "y_pred": y_pred / factor,
            "x": x / factor,
            "y": y / factor,
            "y_pred_times_factor": y_pred,
            "y_times_factor": y,
            "ypred_first": np.vstack([y[0], y_pred[0]]),
        }

        return infer_dict

    engine = Engine(_inference)

    engine.logger = setup_logger(name=name)

    metrics = {
        "loss": Loss(
            loss_fn,
            output_transform=lambda infer_dict: (
                infer_dict["y_pred_times_factor"],
                infer_dict["y_times_factor"],
            ),
        ),
        "mape": mape_metric,
    }

    for name, metric in metrics.items():
        print("attaching metric: " + name)
        metric.attach(engine, name)

    return engine


def postprocess_image_to_log(state: Dict):
    """Sample function to show postprocessing done at log time """
    state.output["im"] = (
        (inverse_mnist_preprocess(state.output["x"])[0] * 255)
        .type(torch.uint8)
        .squeeze()
    )


########################################################################
# Main training loop
########################################################################
@hydra.main(config_path="configs/default.yaml")
def train(cfg: DictConfig) -> None:

    # Determine device (GPU, CPU, etc.)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    model = get_network(cfg)

    # Data Loaders (select example)
    if cfg.example == "classification":
        train_loader, val_loader = get_dataloaders(
            cfg, num_workers=cfg.data_loader_workers, dataset_name="mnist"
        )
        # Your training loop
        trainer = create_classification_training_loop(
            model, cfg, "trainer", device=device
        )
        # Your evaluation loop
        evaluator = create_classification_evaluation_loop(
            model, cfg, "evaluator", device=device
        )
    else:
        train_loader, val_loader = get_dataloaders(
            cfg, num_workers=cfg.data_loader_workers, dataset_name="reunion"
        )
        # Your training loop
        trainer = create_regression_training_loop(model, cfg, "trainer", device=device)
        # Your evaluation loop
        evaluator = create_regression_evaluation_loop(
            model, cfg, "evaluator", device=device
        )

    ld = LogDirector(cfg, engines=[trainer, evaluator])

    # Set configuration defined random seed
    manual_seed(cfg.random_seed)

    ########################################################################
    # Logging Callbacks
    ########################################################################

    # Helper to run the evaluation loop
    def run_evaluator():
        evaluator.run(val_loader)
        # NOTE: Must return the engine we want to log from
        return evaluator

    if cfg.example == "regression":

        ld.set_event_handlers(
            trainer,
            Events.EPOCH_COMPLETED(every=1),
            EngineStateAttr.OUTPUT,
            log_operations=[
                (LOG_OP.LOG_MESSAGE, ["loss"]),
                (LOG_OP.SAVE_IN_DATA_FILE, ["loss"]),
                (
                    LOG_OP.NUMBER_TO_VISDOM,
                    [
                        # First plot, key is "p1"
                        VisPlot(
                            var_name="loss",
                            plot_key="p1",
                            split="mse",
                            env="main",
                            opts={
                                "title": "Train error (loss) ",
                                "x_label": "Iters",
                                "y_label": "nll",
                            },
                        )
                    ],
                ),
            ],
        )

        ld.set_event_handlers(
            trainer,
            Events.EPOCH_COMPLETED(every=20),
            EngineStateAttr.METRICS,
            engine_producer=run_evaluator,
            log_operations=[
                (
                    LOG_OP.NUMBER_TO_VISDOM,
                    [
                        # First plot, key is "p1"
                        VisPlot(
                            var_name="loss",
                            plot_key="p2",
                            split="mse",
                            opts={
                                "title": "Eval error (loss)",
                                "x_label": "Epoch",
                                "y_label": "Loss",
                            },
                            env="main",
                        )
                    ],
                ),
                (
                    LOG_OP.NUMBER_TO_VISDOM,
                    [
                        VisPlot(
                            var_name="mape",
                            plot_key="p3",
                            split="MAPE",
                            opts={
                                "title": "Eval error (MAPE)",
                                "x_label": "Epoch",
                                "y_label": "MAPE",
                            },
                            env="main",
                        )
                    ],
                ),
            ],
        )

        def score_function(engine):
            return evaluator.state.metrics["mape"]

        handler = ModelCheckpoint(
            "models/", "checkpoint", score_function=score_function, n_saved=5
        )

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=50), handler, {"mlp": model}
        )

        ld.set_event_handlers(
            trainer,
            Events.EPOCH_COMPLETED(every=100),
            EngineStateAttr.OUTPUT,
            engine_producer=run_evaluator,
            log_operations=[
                (
                    LOG_OP.VECTOR_TO_VISDOM,
                    [
                        VisPlot(
                            var_name="ypred_first",
                            plot_key="e",
                            split="nll",
                            opts={
                                "title": "Predictions vs. Ground Truth,  # ",
                                "x_label": "Time",
                                "y_label": "Energy Consumption",
                            },
                            env="main",
                        )
                    ],
                )
            ],
        )

    if cfg.example == "classification":
        ld.set_event_handlers(
            trainer,
            Events.ITERATION_COMPLETED(every=50),
            EngineStateAttr.OUTPUT,
            do_before_logging=postprocess_image_to_log,
            log_operations=[
                (LOG_OP.SAVE_IMAGE, ["im"]),  # Save images to a folder
                (LOG_OP.LOG_MESSAGE, ["nll"]),  # Log fields as message in logfile
                (
                    LOG_OP.SAVE_IN_DATA_FILE,
                    ["nll"],
                ),  # Log fields as separate data files
                (
                    LOG_OP.NUMBER_TO_VISDOM,
                    [
                        # First plot, key is "p1"
                        VisPlot(
                            var_name="nll",
                            plot_key="p1",
                            split="nll_1",
                            # Any opts that Visdom supports
                            opts={
                                "title": "Plot 1",
                                "xlabel": "Iters",
                                "fillarea": True,
                            },
                        ),
                        VisPlot(var_name="nll_2", plot_key="p1", split="nll_2"),
                    ],
                ),
                (
                    LOG_OP.IMAGE_TO_VISDOM,
                    [
                        VisImg(
                            var_name="im",
                            img_key="1",
                            env="images",
                            opts={"caption": "a current image", "title": "title"},
                        ),
                        VisImg(
                            var_name="im",
                            img_key="2",
                            env="images",
                            opts={"caption": "a current image", "title": "title"},
                        ),
                    ],
                ),
            ],
        )

        ld.set_event_handlers(
            trainer,
            Events.EPOCH_COMPLETED,
            EngineStateAttr.METRICS,
            # Run the evaluation loop, then do log operations from the return engine
            engine_producer=run_evaluator,
            log_operations=[
                (
                    LOG_OP.LOG_MESSAGE,
                    ["nll", "accuracy"],
                ),  # Log fields as message in logfile
                (
                    LOG_OP.SAVE_IN_DATA_FILE,
                    ["accuracy"],
                ),  # Log fields as separate data files
                (
                    LOG_OP.NUMBER_TO_VISDOM,
                    [
                        # First plot, key is "p1"
                        VisPlot(
                            var_name="accuracy",
                            plot_key="p3",
                            split="acc",
                            # Any opts that Visdom supports
                            opts={"title": "Eval Acc", "xlabel": "Iters"},
                        ),
                        # First plot, key is "p1"
                        VisPlot(
                            var_name="nll",
                            plot_key="p4",
                            split="nll",
                            # Any opts that Visdom supports
                            opts={
                                "title": "Eval Nll",
                                "xlabel": "Iters",
                                "fillarea": True,
                            },
                        ),
                    ],
                ),
            ],
        )

    # Execute training
    if cfg.example == "regression":
        trainer.run(train_loader, max_epochs=1000)
    else:
        trainer.run(train_loader, max_epochs=cfg.mode.train.max_epochs)


if __name__ == "__main__":
    train()
