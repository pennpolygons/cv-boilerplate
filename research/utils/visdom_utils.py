import numpy as np

import os
import time
import random
import cv2

from dataclasses import dataclass
from omegaconf import DictConfig
from visdom import Visdom
from visdom import server
from torchvision.transforms import ToTensor
from torch import Tensor
import matplotlib.pyplot as plt
import matplotlib
import pdb
import signal
import hydra


matplotlib.use("tkagg")


@hydra.main(config_path="../configs", config_name="default.yaml")
def make_visdom(cfg: DictConfig):
    vdb = Visualizer(cfg)


@dataclass
class VisPlotMsg:
    var_name: str
    split_name: str
    title_name: str


class Visualizer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg.visdom

        visdom_command = (
            "screen -S visdom_"
            + str(self.cfg.port)
            + ' -d -m bash -c "python -m visdom.server -port '
            + str(self.cfg.port)
            + '"'
        )

        os.mkdir("visdom")
        os.system(visdom_command)
        time.sleep(2)
        self.env = self.cfg.env_name  # TODO: What is this
        self.vis = Visdom(
            port=self.cfg.port,
            log_to_filename=os.path.join("visdom", self.cfg.log_to_filename),
            offline=self.cfg.offline,
        )
        (self.x_min, self.x_max), (self.y_min, self.y_max) = (
            (self.cfg.x_min, self.cfg.x_max),
            (self.cfg.y_min, self.cfg.y_max),
        )
        self.counter = 0
        self.plots = {}

    def img_result(self, img_list, nrow, caption="view", title="title", win=1):
        self.vis.images(
            img_list, nrow=nrow, win=win, opts={"caption": caption, "title": title}
        )

    def plot_img_255(self, img, caption="view", title="view", win=1):
        self.vis.image(img, win=win, opts={"caption": caption, "title": title})

    def plot_matplotlib(self, fig, caption="view", title="title", win=1):
        self.vis.matplot(
            fig, win=win, opts={"caption": caption, "title": title, "resizable": True}
        )

    def plot_plotly(self, fig, caption="view", title="title", win=1):
        self.vis.plotlyplot(fig, win=win)

    def plot(self, var_name: str, split_name: str, title_name: str, x: int, y: int):
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.line(
                X=np.array([x, x]),
                Y=np.array([y, y]),
                env=self.env,
                opts=dict(
                    legend=[split_name],
                    title=title_name,
                    xlabel="Epochs",
                    ylabel=var_name,
                ),
            )
        else:
            self.vis.line(
                X=np.array([x]),
                Y=np.array([y]),
                env=self.env,
                win=self.plots[var_name],
                name=split_name,
                update="append",
            )


if __name__ == "__main__":
    make_visdom()
