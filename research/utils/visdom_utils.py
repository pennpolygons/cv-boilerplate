import numpy as np

import os
import time
import random
import cv2

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


def make_visdom(cfg: DictConfig):
    vdb = Visualizer(cfg)


class Visualizer:
    def init(self, cfg: DictConfig):

        self.visdom_config = cfg
        visdom_command = (
            "screen -S visdom_"
            + str(visdom_config["port"])
            + ' -d -m bash -c "python -m visdom.server -port '
            + str(visdom_config["port"])
            + '"'
        )
        os.system(visdom_command)
        time.sleep(2)
        self.env = cfg.env_name
        self.vis = visdom.Visdom(
            port=cfg.port,
            log_to_filename="temp/" + cfg.log_to_filename,
            offline=cfg.offline,
        )
        (self.x_min, self.x_max), (self.y_min, self.y_max) = (
            (cfg.x_min, cfg.x_max),
            (cfg.y_min, cfg.y_max),
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

    def plot(self, var_name, split_name, title_name, x, y):
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
