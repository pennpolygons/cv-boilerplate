import logging
import struct
import os
import torch

import numpy as np

from PIL import Image
from enum import Enum
from typing import Callable, List, Dict
from ignite.engine import Events, Engine
from utils.helpers import tern
from utils.visdom_utils import Visualizer, VisPlot, VisImg


def _to_stdout(
    engine: Engine, fields: List[str], engine_attr: str, time_label: str = None,
) -> None:
    """Prints string formatted engine output fields to stdout"""
    log_str = "{} | {}".format(
        time_label,
        "  ".join(
            [
                "{}: {:.3f}".format(field, getattr(engine.state, engine_attr)[field])
                for field in fields
            ]
        ),
    )
    print(log_str)


def _to_log(
    engine: Engine, fields: List[str], engine_attr: str, time_label: str = None,
) -> None:
    """Logs string formatted engine output fields to logfile"""
    log_str = "{} | {}".format(
        time_label,
        "  ".join(
            [
                "{}: {:.3f}".format(field, getattr(engine.state, engine_attr)[field])
                for field in fields
            ]
        ),
    )

    engine.logger.info(log_str)


def _to_file_num(
    engine: Engine, fields: List[str], engine_attr: str, time_label: str
) -> None:
    """Save engine output fields as separate binary data files"""
    for field in fields:
        if not field in engine.state.fp:
            engine.state.fp[field] = open("{}.pt".format(field), "wb+")

        engine.state.fp[field].write(
            struct.pack("f", getattr(engine.state, engine_attr)[field])
        )


def _to_file_img(
    engine: Engine, fields: List[str], engine_attr: str, time_label: str
) -> None:
    """Save engine output fields as images"""
    value_dict = getattr(engine.state, engine_attr)

    for field in fields:

        if not os.path.exists(os.path.join(engine.logger.name, field)):
            os.makedirs(os.path.join(engine.logger.name, field))

        if torch.is_tensor(value_dict[field]):
            im = Image.fromarray(
                value_dict[field].detach().cpu().numpy().astype(np.uint8)
            )
        else:
            im = Image.fromarray(value_dict[field].astype(np.uint8))

        im.save(
            os.path.join(
                engine.logger.name,
                field,
                "{:03d}_{:06d}_{}.png".format(
                    engine.state.epoch, engine.state.iteration, field
                ),
            )
        )


def _number_to_visdom(
    engine: Engine,
    vis: Visualizer,
    vis_plot_msgs: List[VisPlot],
    engine_attr: str,
    **kwargs
    # time_label: int,
) -> None:
    """Log numeric engine output to Visdom server"""
    for msg in vis_plot_msgs:
        vis.plot(
            msg.plot_key,
            msg.split,
            kwargs["time_label"],
            getattr(engine.state, engine_attr)[msg.var_name],
            env=msg.env,
            opts=msg.opts,
        )

"""
def _vector_to_visdom(
    engine: Engine,
    vis: Visualizer,
    vis_plot_msgs: List[VisPlot],
    engine_attr: str,
    x_value: int = None,
    epoch_num: int = None,
    **kwargs,
) -> None:
    """Save engine output to Visdom server"""
    value_dict = getattr(engine.state, engine_attr)

    if x_value is None:

        x_value = np.arange(len(value_dict[msg.var_name]))

    for msg in vis_plot_msgs:

        vis.plot_line(
            msg.plot_key,
            msg.split,
            x_value,
            value_dict[msg.var_name],
            env=msg.env,
            opts=msg.opts,
        )"""


def _image_to_visdom(
    engine: Engine,
    vis: Visualizer,
    vis_img_msgs: List[VisImg],
    engine_attr: str,
    **kwargs
) -> None:
    """Log image engine output to Visdom server"""
    for msg in vis_img_msgs:
        vis.plot_img_255(
            msg.img_key,
            getattr(engine.state, engine_attr)[msg.var_name],
            env=msg.env,
            opts=msg.opts,
        )


class LOG_OP(Enum):
    """Enum wrapper around logging modes"""

    # Log to standard out (print())
    PRINT = _to_stdout
    # Log as message in log file
    LOG_MESSAGE = _to_log
    # Log to separate file
    SAVE_IN_DATA_FILE = _to_file_num
    # Save image to standalone folder
    SAVE_IMAGE = _to_file_img
    # Log to visdom
    NUMBER_TO_VISDOM = _number_to_visdom
    # Log image to visdom
    IMAGE_TO_VISDOM = _image_to_visdom
