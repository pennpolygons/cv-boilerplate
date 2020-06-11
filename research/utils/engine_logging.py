from enum import Enum
from typing import Callable, List, Dict
from ignite.engine import Engine
from torch.utils.data import DataLoader


def _to_stdout(engine: Engine, fields: List[str], epoch_num=None) -> None:
    """Prints string formatted engine output fields to stdout"""
    log_str = "  ".join(
        ["{}: {:.3f}".format(field, engine.state.output[field]) for field in fields]
    )
    log_str = "Epoch[{:d}] {}".format(tern(epoch_num, engine.state.epoch), log_str)
    print(log_str)


def _to_log(engine: Engine, fields: List[str], epoch_num=None) -> None:
    """Logs string formatted engine output fields to logfile"""
    log_str = "  ".join(
        ["{}: {:.3f}".format(field, engine.state.output[field]) for field in fields]
    )
    log_str = "Epoch[{:d}] {}".format(tern(epoch_num, engine.state.epoch), log_str)
    engine.logger.info(log_str)


def _to_file(engine: Engine, fields: List[str]) -> None:
    """Logs formatted engine output fields to separate file (binary)"""
    for field in fields:
        # Check that engine has fp in state
        # Check
        pass


class LOG_MODE(Enum):
    # Log to standard out (print())
    STDOUT = _to_stdout
    # Log as message in log file
    LOG_MSG = _to_log
    # Log to separate file
    LOG_FIL = _to_file
    # Log image to standalone folder
    LOG_IMG = 3
    # Log to visdom
    VISDOM_NUM = 4
    # Log image to visdom
    VISDOM_IMG = 5


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


# FIXME: Engine doesn't seem to have "times" attribute in engine state contrary to docs
# def log_total_time(engine: Engine) -> None:
#     """Log the total time to complete training"""
#     engine.logger.info("Total: {}".format(engine.state.times["COMPLETED"]))


def log_engine_output(
    engine: Engine, fields: Dict[LOG_MODE, List[str]], epoch_num=None
) -> None:
    """Log numerical fields in the engine output dictionary to stdout"""
    import pdb

    pdb.set_trace()
    for mode in fields.keys():
        # mode((engine, fields[mode], epoch_num)) confirm this
        pass


def log_engine_metrics(
    engine: Engine, loader: DataLoader, fields: List[str], epoch_num=None,
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
