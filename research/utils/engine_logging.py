from typing import Callable, List
from ignite.engine import Engine


def _lf(log_fn: Callable[[Engine, List[str]], None], fields: List[str]):
    """Returns a log function lambda. Useful for code clarity"""
    return lambda _: log_fn(_, fields=fields)


def log_fields_stdout(engine: Engine, fields: List[str]) -> None:
    """Log several fields in the engine output dictionary to stdout"""
    log_str = "  ".join(
        ["{}: {:.2f}".format(field, engine.state.output[field]) for field in fields]
    )
    print("Epoch[{}] {}".format(engine.state.epoch, log_str))


# FIXME: Use logging module, not print statements
def log_training_loss(engine: Engine, field: str = "loss") -> None:
    print(
        "Epoch[{}] Loss: {:.2f}".format(engine.state.epoch, engine.state.output[field])
    )


# FIXME: Use logging module, not print statements
def log_training_results(engine: Engine) -> None:
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print(
        "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
            engine.state.epoch, metrics["accuracy"], metrics["nll"]
        )
    )


# FIXME: Use logging module, not print statements
def log_validation_results(engine: Engine) -> None:
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(
        "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
            engine.state.epoch, metrics["accuracy"], metrics["nll"]
        )
    )


# TODO: Add Visdom logging callbacks
# TODO: Add Slack notification callback
