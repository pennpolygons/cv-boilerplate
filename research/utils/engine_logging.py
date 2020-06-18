from typing import Callable, List, Dict, Any, Union, Tuple
from ignite.engine import Engine
from torch.utils.data import DataLoader
from utils.log_operations import LOG_OP
from utils.helpers import tern
from utils.visdom_utils import VisPlot, VisImg

# Defining a custom type for clarity
LOG_OP_ARGS = Union[List[str], List[VisPlot], List[VisImg]]


def _lf_one(
    log_fn: Callable[[Engine, List[str]], None],
    log_ops: List[Tuple[LOG_OP, LOG_OP_ARGS]],
    **kwargs
) -> Callable[[Engine], Any]:
    """Returns a lambda calling custom log function with one engine (e.g. the training loop)"""
    return lambda engine: log_fn(
        engine, log_ops, epoch_num=engine.state.epoch, **kwargs
    )


def _lf_two(
    log_fn: Callable[[Engine, LOG_OP_ARGS], None],
    inner_engine: Engine,
    loader: DataLoader,
    log_ops: List[Tuple[LOG_OP, LOG_OP_ARGS]],
    **kwargs
) -> Callable[[Engine], Any]:
    """Returns a lambda calling custom log function with two engines (e.g. the training loop and validation loop)"""
    return lambda outer_engine: log_fn(
        inner_engine, loader, log_ops, epoch_num=outer_engine.state.epoch, **kwargs
    )


def log_engine_output(
    engine: Engine, log_ops: List[Tuple[LOG_OP, LOG_OP_ARGS]], epoch_num=None,
) -> None:
    """Log numerical fields in the engine output dictionary to stdout"""
    for op, op_args in log_ops:
        if op is LOG_OP.NUMBER_TO_VISDOM or op is LOG_OP.IMAGE_TO_VISDOM:
            op(engine, engine.state.vis, op_args, engine_attr="output")
        else:
            op(engine, op_args, engine_attr="output")


def run_engine_and_log_metrics(
    engine: Engine,
    loader: DataLoader,
    log_ops: List[Tuple[LOG_OP, LOG_OP_ARGS]],
    epoch_num=None,
) -> None:

    """Run engine on Dataloader. Then log numerical fields in the engine metrics dictionary to stdout"""
    engine.run(loader)
    for op, op_args in log_ops:
        if op is LOG_OP.NUMBER_TO_VISDOM or op is LOG_OP.IMAGE_TO_VISDOM:
            op(engine, engine.state.vis, op_args, engine_attr="metrics")
        else:
            op(engine, op_args, engine_attr="metrics")


# TODO: Add Slack notification callback

# FIXME: Engine doesn't seem to have "times" attribute in engine state contrary to docs
# def log_total_time(engine: Engine) -> None:
#     """Log the total time to complete training"""
#     engine.logger.info("Total: {}".format(engine.state.times["COMPLETED"]))
