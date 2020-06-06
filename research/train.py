import hydra

from omegaconf import DictConfig
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss

from dataset import get_dataloaders
from networks import get_network

import pdb


def create_supervised_trainer(cfg: DictConfig, device=None) -> Engine:

    # Network
    model = get_network(cfg)
    # Data Loader
    train_loader, val_loader = get_dataloaders(cfg)
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    # Loss
    loss = torch.nn.NLLLoss()

    def _update(engine, batch):

        ########################################################################
        # Modify the scientific business logic of your training loop
        ########################################################################

        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device)
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

    # Training loop Logic
    trainer = create_supervised_trainer(cfg, device=None)

    # Callbacks
    trainer.add_event_handler(Events.STARTED, lambda engine: print("Start training"))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss)

    # train()
    trainer.run(train_loader, max_epochs=config.train.max_epochs)


if __name__ == "__main__":
    train()
