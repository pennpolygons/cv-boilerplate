from typing import Callable, List, Dict, Any, Union, Tuple
from ignite.engine import Engine
from torch.utils.data import DataLoader
from utils.log_operations import LOG_OP
from utils.helpers import tern
from utils.visdom_utils import VisPlot, VisImg
from enum import Enum

# Defining a custom type for clarity
LOG_OP_ARGS = Union[List[str], List[VisPlot], List[VisImg]]


# TODO: Add Slack notification callback

# FIXME: Engine doesn't seem to have "times" attribute in engine state contrary to docs
# def log_total_time(engine: Engine) -> None:
#     """Log the total time to complete training"""
#     engine.logger.info("Total: {}".format(engine.state.times["COMPLETED"]))
