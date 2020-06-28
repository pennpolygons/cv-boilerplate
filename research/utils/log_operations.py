import logging
import struct, pdb, os
import numpy as np
import torch

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
    value_dict = getattr(engine.state, engine_attr)

    log_str = "  ".join(
        ["{}: {:.3f}".format(field, value_dict[field]) for field in fields]
    )
    log_str = "{} | {}".format(time_label, log_str,)
    print(log_str)


def _to_log(
    engine: Engine, fields: List[str], engine_attr: str, time_label: str = None,
) -> None:
    """Logs string formatted engine output fields to logfile"""
    value_dict = getattr(engine.state, engine_attr)

    log_str = "  ".join(
        ["{}: {:.3f}".format(field, value_dict[field]) for field in fields]
    )
    log_str = "{} | {}".format(time_label, log_str,)
    engine.logger.info(log_str)


def _to_file(
    engine: Engine, fields: List[str], engine_attr: str, time_label: str
) -> None:
    """Save engine output fields as separate binary data files"""
    value_dict = getattr(engine.state, engine_attr)

    for field in fields:
        if not field in engine.state.fp:
            engine.state.fp[field] = open("{}.pt".format(field), "wb+")

        engine.state.fp[field].write(struct.pack("f", value_dict[field]))


def _to_img(
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
    engine: Engine, vis: Visualizer, vis_plot_msgs: List[VisPlot], engine_attr: str,
) -> None:
    """Save engine output to Visdom server"""
    value_dict = getattr(engine.state, engine_attr)

    for msg in vis_plot_msgs:
        vis.plot(
            msg.plot_key,
            msg.split,
            msg.title,
            x_value,
            value_dict[msg.var_name],
            x_label=msg.x_label,
            y_label=msg.y_label,
            env=msg.env,
        )


def _image_to_visdom(
    engine: Engine, vis: Visualizer, vis_img_msgs: List[VisImg], engine_attr: str,
) -> None:
    value_dict = getattr(engine.state, engine_attr)
    for msg in vis_img_msgs:
        vis.plot_img_255(
            value_dict[msg.var_name], caption=msg.caption, title=msg.title, env=msg.env,
        )


class LOG_OP(Enum):
    """Enum wrapper around logging modes"""

    # Log to standard out (print())
    PRINT = _to_stdout
    # Log as message in log file
    LOG_MESSAGE = _to_log
    # Log to separate file
    SAVE_IN_DATA_FILE = _to_file
    # Save image to standalone folder
    SAVE_IMAGE = _to_img
    # Log to visdom
    NUMBER_TO_VISDOM = _number_to_visdom
    # Log image to visdom
    IMAGE_TO_VISDOM = _image_to_visdom
