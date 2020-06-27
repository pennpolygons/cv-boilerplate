from omegaconf import DictConfig
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Engine, Events

from utils.visdom_utils import Visualizer, VisPlot, VisImg
from utils.engine_logging import _lf_one, _lf_switch
from utils.log_operations import LOG_OP

from typing import Callable, List, Dict, Any, Union, Tuple

# Defining a custom type for clarity
LOG_OP_ARGS = Union[List[str], List[VisPlot], List[VisImg]]


class LogDirector:
    pass

    def __init__(self, cfg: DictConfig, engines: List[Engine] = None):
        # TODO: Set up a Tensorboard contrib.handler
        self.tb_writer = None

        self.registered_engines = {}

        # Spin up Visdom server
        self.vis = Visualizer(cfg)

        if engines:
            self.register_engines(engines)

    def startup_engine(self, engine: Engine) -> None:
        engine.state.fp = {}
        engine.state.vis = self.vis

    def register_engines(self, engines: List[Engine]) -> None:
        for eng in engines:
            eng.add_event_handler(Events.STARTED, self.startup_engine)
            self.registered_engines[eng.logger.name] = eng

    def print_event_handlers(self):
        # TODO: Should pretty print all the event handlers
        raise NotImplementedError()

    # Set's up a bunch of log handlers
    def set_event_handlers(
        self,
        engine: Engine,
        event: Events,
        engine_attr: str,
        log_operations: List[Tuple[LOG_OP, LOG_OP_ARGS]],
        pre_op: Callable[[Any], Engine] = None,
    ):
        # Log operations on the calling engine

        engine.add_event_handler(
            event,
            lambda engine: [
                log_op(
                    engine,
                    op_args,
                    engine_attr=engine_attr,
                    epoch_num=engine.state.epoch,
                )
                for log_op, op_args in log_operations
            ],
        )

