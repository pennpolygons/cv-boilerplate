
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
from utils.visdom_utils import Visualizer, VisPlot, VisImg
from utils.image_utils import inverse_mnist_preprocess

from utils.engine_logging import (
    _lf_one,
    _lf_two,
    log_engine_output,
    run_engine_and_log_metrics,
    LOG_OP,
)


def startup_engine(engine: Engine, vis: Visualizer = None) -> None:
    engine.state.fp = {}
    engine.state.vis = vis


def create_training_loop(
    model: nn.Module, cfg: DictConfig, name: str, device="cpu"
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
        update_dict = {
            "nll": loss.item(),
            "nll_2": loss.item() + 0.5,
            "y_pred": y_pred,
            "y": y,
            "im": (inverse_mnist_preprocess(x)[0] * 255).type(torch.uint8).squeeze()
        }

        return update_dict

    engine = Engine(_update)

    # Required to set up logging
    engine.logger = setup_logger(name=name)

    # (Optional) Specify training metrics. "output_transform" used to select items from "update_dict" needed by metrics
    # Collecting metrics over training set is not recommended
    # https://pytorch.org/ignite/metrics.html#ignite.metrics.Loss

    return engine


def create_evaluation_loop(
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
        infer_dict = {
            "y_pred": y_pred,
            "y": y
        }

        return infer_dict

    engine = Engine(_inference)

    # Required to set up logging
    engine.logger = setup_logger(name=name)

    # Specify evaluation metrics. "output_transform" used to select items from "infer_dict" needed by metrics
    # https://pytorch.org/ignite/metrics.html#ignite.metrics.
    metrics = {
        "accuracy": Accuracy(output_transform=lambda infer_dict: (infer_dict["y_pred"], infer_dict["y"])),
        "nll": Loss(loss_fn, output_transform=lambda infer_dict: (infer_dict["y_pred"], infer_dict["y"])),
    }

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

    # Spin up visdom
    vis = Visualizer(cfg)

    # Model
    model = get_network(cfg)

    # Data Loaders
    train_loader, val_loader = get_dataloaders(cfg, num_workers=cfg.data_loader_workers)

    # Your training loop
    trainer = create_training_loop(model, cfg, "trainer", device=device)    
    # Your evaluation loop
    evaluator = create_evaluation_loop(model, cfg, "evaluator", device=device)
    
    ########################################################################
    # Logging Callbacks
    ########################################################################

    #!!!! Required. Do not change. !!!!#
    trainer.add_event_handler(Events.STARTED, lambda _: startup_engine(_, vis=vis))
    evaluator.add_event_handler(Events.STARTED, lambda _: startup_engine(_, vis=vis))


    # Perform various log operations on the "trainer" engine output every 50 iterations
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=50),
        # The function _lf_one() is required to pass the "trainer" engine to "log_engine_output"
        _lf_one(
            log_engine_output,
            {
                LOG_OP.SAVE_IMAGE: ["im"],                                          # Save image to folder
                LOG_OP.LOG_MESSAGE: ["nll"],                                        # Log fields as message in logfile
                LOG_OP.SAVE_IN_DATA_FILE: ["nll"],                                  # Log fields as separate data files
                LOG_OP.IMAGE_TO_VISDOM: [
                    VisImg("im", caption="caption", title="title")                  # Display image in Visdom
                ],      
                LOG_OP.NUMBER_TO_VISDOM: [                                          # Plot fields to Visdom
                    VisPlot("nll", plot_key="p1", split="nll_1", title="Plot 1"),
                    VisPlot("nll_2", plot_key="p1", split="nll_2"),
                    VisPlot("nll", plot_key="p2", split="nll", title="Plot 2"),
                ] 
            },
        ),
    )

    # Perform various log operations on metrics collected in the "evaluator" engine output every epoch
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        # The function _lf_two() is required to pass the "trainer" and "evaluator" engines to "run_engine_and_log_metrics"
        _lf_two(
            # Run the "evaluator" engine (i.e. evaluation loop) on "val_loader" and log metrics
            run_engine_and_log_metrics,
            evaluator,
            val_loader,
            {
                LOG_OP.LOG_MESSAGE: ["nll", "accuracy"],    # Log fields as message in logfile
                LOG_OP.SAVE_IN_DATA_FILE: ["accuracy"],     # Log fields as separate data files
            },
        ),
    )

    # Execute training
    trainer.run(train_loader, max_epochs=cfg.mode.train.max_epochs)


if __name__ == "__main__":
    train()
