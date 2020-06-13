import logging
import struct, pdb, os
import numpy as np

from PIL import Image
from enum import Enum
from typing import Callable, List, Dict
from ignite.engine import Events, Engine
from utils.helpers import tern
from utils.image_utils import convert_cwh_to_whc


def _to_stdout(
    engine: Engine, fields: List[str], engine_attr: str, epoch_num=None, iter_num=None
) -> None:
    """Prints string formatted engine output fields to stdout"""
    value_dict = getattr(engine.state, engine_attr)

    log_str = "  ".join(
        ["{}: {:.3f}".format(field, value_dict[field]) for field in fields]
    )
    log_str = "Epoch[{:d}], Iter[{:d}] | {}".format(
        tern(epoch_num, engine.state.epoch),
        tern(iter_num, engine.state.iteration),
        log_str,
    )
    print(log_str)


def _to_log(
    engine: Engine, fields: List[str], engine_attr: str, epoch_num=None, iter_num=None,
) -> None:
    """Logs string formatted engine output fields to logfile"""
    value_dict = getattr(engine.state, engine_attr)

    log_str = "  ".join(
        ["{}: {:.3f}".format(field, value_dict[field]) for field in fields]
    )
    log_str = "Epoch[{:d}], Iter[{:d}] | {}".format(
        tern(epoch_num, engine.state.epoch),
        tern(iter_num, engine.state.iteration),
        log_str,
    )
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
        if not os.path.exists(field):
            os.mkdir(field)

        im = Image.fromarray(value_dict[field].numpy().astype(np.uint8))
        im.save(
            os.path.join(
                field,
                "{:03d}_{:06d}_{}.png".format(
                    engine.state.epoch, engine.state.iteration, field
                ),
            )
        )


class LOG_OP(Enum):
    """Enum wrapper around logging modes"""

    # Log to standard out (print())
    PRINT = _to_stdout
    # Log as message in log file
    LOG_MESSAGE = _to_log
    # Log to separate file
    SAVE_IN_DATA_FILE = _to_file
    # Log image to standalone folder
    SAVE_IMAGE = _to_img

    # TODO: Add logging operations for Visdom

    # Log to visdom
    # NUMBER_TO_VISDOM = ...
    # Log image to visdom
    # IMAGE_TO_VISDOM = ....
