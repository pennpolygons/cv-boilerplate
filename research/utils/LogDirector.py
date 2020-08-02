import logging

from enum import Enum
from omegaconf import DictConfig
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Engine, Events
from ignite.utils import setup_logger

from utils.visdom_utils import Visualizer, VisPlot, VisImg
from utils.log_operations import LOG_OP

from typing import Callable, List, Dict, Any, Union, Tuple

# Defining a custom type for clarity
LOG_OP_ARGS = Union[List[str], List[VisPlot], List[VisImg]]


class LogTimeLabel(Enum):
    @classmethod
    def global_iteration(cls, engine: Engine):
        return engine.state.epoch_length * engine.state.epoch + engine.state.iteration

    CUR_ITER_IN_EPOCH = lambda engine: {
        "int_label": LogTimeLabel.global_iteration(engine),
        "str_label": "Epoch[{:d}], Iter[{:d}]".format(
            engine.state.epoch, engine.state.iteration
        ),
    }

    GLOBAL_ITER = lambda engine: {
        "int_label": LogTimeLabel.global_iteration(engine),
        "str_label": "Global Iter[{:d}]".format(
            engine.state.epoch_length * engine.state.epoch + engine.state.iteration
        ),
    }

    CUR_EPOCH = lambda engine: {
        "int_label": engine.state.epoch,
        "str_label": "Epoch[{:d}]".format(engine.state.epoch),
    }


class EngineStateAttr(Enum):
    METRICS = "metrics"  # engine.state.metrics
    OUTPUT = "output"  # engine.state.output


class LogDirector:
    def __init__(self, cfg: DictConfig, engines: List[Engine] = None):
        # TODO: Set up a Tensorboard contrib.handler
        self.tb_writer = None
        self.registered_engines = {}
        self.log_file = cfg.files.log_file

        # Spin up Visdom server
        self.vis = Visualizer(cfg)

        if engines:
            self.register_engines(engines)

    def startup_engine(self, engine: Engine) -> None:
        engine.state.fp = {}
        engine.state.vis = self.vis

    def register_engines(self, engines: List[Engine]) -> None:
        print("Ladies and gentlemen... start your engines!")
        for eng in engines:
            eng.add_event_handler(Events.STARTED, self.startup_engine)
            self.registered_engines[eng.logger.name] = eng

            eng.logger = setup_logger(
                name=eng.logger.name, level=logging.INFO, filepath=self.log_file
            )

    def print_event_handlers(self):
        # TODO: Should pretty print all the event handlers
        raise NotImplementedError()

    # Set's up a bunch of log handlers
    def set_event_handlers(
        self,
        engine: Engine,
        event: Events,
        engine_attr: EngineStateAttr,
        log_time_label: LogTimeLabel = LogTimeLabel.CUR_ITER_IN_EPOCH,
        engine_producer: Callable[[Any], Engine] = None,
        # Function to run before log operations. Mutate engine state output
        do_before_logging: Callable[[Dict], Any] = None,
        # The log operations (save-to-file, save-to-visdom, etc.)
        log_operations: List[Tuple[LOG_OP, LOG_OP_ARGS]] = [],
    ):
        def _do_all_log_operations(engine: Engine):
            engine_to_log_from = engine_producer() if engine_producer else engine

            if do_before_logging:
                do_before_logging(engine_to_log_from.state)

            for log_op, op_args in log_operations:
                if (
                    log_op is LOG_OP.NUMBER_TO_VISDOM
                    or log_op is LOG_OP.IMAGE_TO_VISDOM
                    or log_op is LOG_OP.VECTOR_TO_VISDOM
                ):
                    visdom_log_op = log_op
                    visdom_log_op(
                        engine_to_log_from,
                        self.vis,
                        op_args,
                        engine_attr=engine_attr.value,
                        # Always use event attached engine for time_label
                        time_label=log_time_label(engine)["int_label"],
                    )
                else:
                    log_op(
                        engine_to_log_from,
                        op_args,
                        engine_attr=engine_attr.value,
                        # Always use event attached engine for time_label
                        time_label=log_time_label(engine)["str_label"],
                    )

        # Bind the callables list to the event handler
        engine.add_event_handler(event, _do_all_log_operations)
