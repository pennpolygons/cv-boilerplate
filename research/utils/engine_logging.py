from typing import Callable, List
from ignite.engine import Engine
from torch.utils.data import DataLoader


def _lf(log_fn: Callable[[Engine, List[str]], None], fields: List[str], **kwargs):
    """Returns a log function lambda. Useful for code clarity"""
    return lambda _: log_fn(_, fields, **kwargs)


def _lf_val(
    log_fn: Callable[[Engine, List[str]], None],
    engine: Engine,
    loader: DataLoader,
    fields: List[str],
    **kwargs
):
    """Returns a log function lambda. Useful for code clarity"""
    return lambda _: log_fn(_, engine, loader, fields, **kwargs)


# FIXME: Engine doesn't seem to have "times" attribute in engine state contrary to docs
# def log_total_time(engine: Engine) -> None:
#     """Log the total time to complete training"""
#     engine.logger.info("Total: {}".format(engine.state.times["COMPLETED"]))


def log_engine_output(engine: Engine, fields: List[str], stdout=False) -> None:
    """Log numerical fields in the engine output dictionary to stdout"""
    log_str = "  ".join(
        ["{}: {:.2f}".format(field, engine.state.output[field]) for field in fields]
    )
    engine.logger.info(log_str)

    if stdout:
        print("Epoch[{}] {}".format(engine.state.epoch, log_str))


def log_engine_metrics(
    train_engine: Engine,
    eval_engine: Engine,
    loader: DataLoader,
    fields: List[str],
    stdout=False,
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
    eval_engine.logger.info(log_str)

    if stdout:
        print("Epoch[{}] {}".format(train_engine.state.epoch, log_str))


# TODO: Add Visdom logging callbacks
# TODO: Add Slack notification callback
