from typing import Callable, List
from ignite.engine import Engine
from torch.utils.data import DataLoader


def _lf(log_fn: Callable[[Engine, List[str]], None], fields: List[str], **kwargs):
    """Returns a log function lambda. Useful for code clarity"""
    return lambda engine: log_fn(engine, fields, epoch_num=engine.state.epoch, **kwargs)


def _lf_val(
    log_fn: Callable[[Engine, List[str]], None],
    val_engine: Engine,
    loader: DataLoader,
    fields: List[str],
    **kwargs
):
    """Returns a log function lambda. Useful for code clarity"""
    return lambda train_engine: log_fn(
        val_engine, loader, fields, epoch_num=train_engine.state.epoch, **kwargs
    )


# FIXME: Engine doesn't seem to have "times" attribute in engine state contrary to docs
# def log_total_time(engine: Engine) -> None:
#     """Log the total time to complete training"""
#     engine.logger.info("Total: {}".format(engine.state.times["COMPLETED"]))


def log_engine_output(
    engine: Engine, fields: List[str], stdout=False, epoch_num=None
) -> None:
    """Log numerical fields in the engine output dictionary to stdout"""
    log_str = "  ".join(
        ["{}: {:.3f}".format(field, engine.state.output[field]) for field in fields]
    )
    log_str = "Epoch[{:d}] {}".format(tern(epoch_num, engine.state.epoch), log_str)
    engine.logger.info(log_str)

    if stdout:
        print(log_str)


def log_engine_metrics(
    engine: Engine, loader: DataLoader, fields: List[str], stdout=False, epoch_num=None,
) -> None:

    """Run engine on Dataloader. Then log numerical fields in the engine metrics dictionary to stdout"""
    engine.run(loader)
    metrics = engine.state.metrics
    log_str = "  ".join(
        ["{}: {:.3f}".format(field, engine.state.metrics[field]) for field in fields]
    )
    log_str = "Epoch[{:d}] {}".format(tern(epoch_num, engine.state.epoch), log_str)
    engine.logger.info(log_str)

    if stdout:
        print(log_str)


# TODO: Add Visdom logging callbacks
# TODO: Add Slack notification callback
