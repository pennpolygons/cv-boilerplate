import hydra
import torch

from omegaconf import DictConfig
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss

from dataset import get_dataloaders
from networks import get_network

from utils.logging import log_training_loss, log_training_results

import pdb


def create_supervised_trainer(cfg: DictConfig, device="cpu") -> Engine:

    # Network
    model = get_network(cfg)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    # Loss
    loss_fn = torch.nn.NLLLoss()

    metrics = {"accuracy": Accuracy(), "nll": Loss(loss_fn)}

    def _update(engine, batch):

        ########################################################################
        # Modify the logic of your training loop
        ########################################################################
        model.train()
        optimizer.zero_grad()
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item(), y_pred, y

    def _metrics_transform(output):
        return output[1], output[2]

    engine = Engine(_update)

    for name, metric in metrics.items():
        metric._output_transform = _metrics_transform
        metric.attach(engine, name)

    return engine


########################################################################
# Main training loop
########################################################################
@hydra.main(config_path="configs/default.yaml")
def train(cfg: DictConfig) -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data Loader
    train_loader, val_loader = get_dataloaders(cfg)

    # Training loop logic
    trainer = create_supervised_trainer(cfg, device=device)

    # Callbacks
    trainer.add_event_handler(Events.STARTED, lambda engine: print("Start training"))
    # trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss)

    # train()
    trainer.run(train_loader, max_epochs=cfg.mode.train.max_epochs)


if __name__ == "__main__":
    train()
