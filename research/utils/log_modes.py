import logging
import struct, pdb

from enum import Enum
from typing import Callable, List, Dict
from ignite.engine import Events, Engine
from utils.helpers import tern


def _to_stdout(
    engine: Engine, fields: List[str], engine_attr: str, epoch_num=None
) -> None:
    """Prints string formatted engine output fields to stdout"""
    value_dict = getattr(engine.state, engine_attr)

    log_str = "  ".join(
        ["{}: {:.3f}".format(field, value_dict[field]) for field in fields]
    )
    log_str = "Epoch[{:d}] {}".format(tern(epoch_num, engine.state.epoch), log_str)
    print(log_str)


def _to_log(
    engine: Engine, fields: List[str], engine_attr: str, epoch_num=None
) -> None:
    """Logs string formatted engine output fields to logfile"""
    value_dict = getattr(engine.state, engine_attr)

    log_str = "  ".join(
        ["{}: {:.3f}".format(field, value_dict[field]) for field in fields]
    )
    log_str = "Epoch[{:d}] {}".format(tern(epoch_num, engine.state.epoch), log_str)
    pdb.set_trace()
    engine.logger.info(log_str)


def _to_file(engine: Engine, fields: List[str], engine_attr: str) -> None:
    """Logs formatted engine output fields to separate file (binary)"""
    value_dict = getattr(engine.state, engine_attr)

    for field in fields:
        # TODO: Should be done exactly once at engine startup
        if not field in engine.state.fp:
            engine.state.fp[field] = open("{}.pt".format(field), "wb+")

        engine.state.fp[field].write(struct.pack("f", value_dict[field]))


def _to_img(engine: Engine, fields: List[str], engine_attr: str) -> None:
    """Logs formatted engine output fields to separate file (binary)"""
    value_dict = getattr(engine.state, engine_attr)

    for field in fields:
        # TODO: Should be done exactly once at engine startup
        if not field in engine.state.fp:
            engine.state.fp[field] = open("{}.pt".format(field), "wb+")

        engine.state.fp[field].write(struct.pack("f", value_dict[field]))


class LOG_MODE(Enum):
    """Enum wrapper around logging modes"""

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
