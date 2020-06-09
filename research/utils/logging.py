from typing import Callable, List
from ignite.engine import Engine
from torch.utils.data import DataLoader


def _lf(log_fn: Callable[[Engine, List[str]], None], fields: List[str]):
    """Returns a log function lambda. Useful for code clarity"""
    return lambda _: log_fn(_, fields=fields)


def _lf_val(
    log_fn: Callable[[Engine, List[str]], None],
    engine: Engine,
    loader: DataLoader,
    fields: List[str],
):
    """Returns a log function lambda. Useful for code clarity"""
    return lambda _: log_fn(_, engine, loader, fields=fields)


def log_engine_output_stdout(engine: Engine, fields: List[str]) -> None:
    """Log numerical fields in the engine output dictionary to stdout"""
    log_str = "  ".join(
        ["{}: {:.2f}".format(field, engine.state.output[field]) for field in fields]
    )
    print("Epoch[{}] {}".format(engine.state.epoch, log_str))


def log_engine_metrics_stdout(
    train_engine: Engine, eval_engine: Engine, loader: DataLoader, fields: List[str]
) -> None:
    """Run eval_engine on Dataloader. Then log numerical fields in the engine metrics dictionary to stdout"""
    eval_engine.run(loader)
    metrics = eval_engine.state.metrics
    log_str = "  ".join(
        [
            "{}: {:.2f}".format(field, eval_engine.state.metrics[field])
            for field in fields
        ]
    )

    print("Epoch[{}] {}".format(train_engine.state.epoch, log_str))


# TODO: Add Visdom logging callbacks
# TODO: Add Slack notification callback
