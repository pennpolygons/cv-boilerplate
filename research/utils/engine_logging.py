import pdb
from enum import Enum
from typing import Callable, List, Dict
from ignite.engine import Engine
from torch.utils.data import DataLoader
from utils.log_modes import LOG_MODE
from utils.helpers import tern


def _lf(
    log_fn: Callable[[Engine, List[str]], None],
    fields: Dict[LOG_MODE, List[str]],
    **kwargs
):
    """Returns a log function lambda. Useful for code clarity"""
    return lambda engine: log_fn(engine, fields, epoch_num=engine.state.epoch, **kwargs)


def _lf_val(
    log_fn: Callable[[Engine, List[str]], None],
    val_engine: Engine,
    loader: DataLoader,
    fields: Dict[LOG_MODE, List[str]],
    **kwargs
):
    """Returns a log function lambda. Useful for code clarity"""
    return lambda train_engine: log_fn(
        val_engine, loader, fields, epoch_num=train_engine.state.epoch, **kwargs
    )


def log_engine_output(
    engine: Engine, fields: Dict[LOG_MODE, List[str]], epoch_num=None
) -> None:
    """Log numerical fields in the engine output dictionary to stdout"""
    pdb.set_trace()
    for mode in fields.keys():
        mode(engine, fields[mode], "output")


def log_engine_metrics(
    engine: Engine, loader: DataLoader, fields: List[str], epoch_num=None,
) -> None:

    """Run engine on Dataloader. Then log numerical fields in the engine metrics dictionary to stdout"""
    engine.run(loader)
    for mode in fields.keys():
        mode(engine, fields[mode], "metrics")


# TODO: Add Visdom logging callbacks
# TODO: Add Slack notification callback

# FIXME: Engine doesn't seem to have "times" attribute in engine state contrary to docs
# def log_total_time(engine: Engine) -> None:
#     """Log the total time to complete training"""
#     engine.logger.info("Total: {}".format(engine.state.times["COMPLETED"]))
