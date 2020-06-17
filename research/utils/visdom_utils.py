
import os
import time
import hydra
import matplotlib

import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
from omegaconf import DictConfig
from visdom import server, Visdom


matplotlib.use("tkagg")


@hydra.main(config_path="../configs", config_name="default.yaml")
def make_visdom(cfg: DictConfig):
    vdb = Visualizer(cfg)


@dataclass
class VisPlot:
    var_name: str # Field name in the engine state (i.e. engine.outputs.var_name)
    plot_key: str # Plot used to plot data. Non-existant plot key creates new plot 
    split: str # Legend split
    title: str = None # Title
    x_label: str = None
    y_label: str = None

@dataclass
class VisImg:
    var_name: str
    caption: str = None
    title: str = None


class Visualizer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg.visdom

        visdom_command = (
            'screen -S visdom_{} -d -m bash -c "python -m visdom.server -port {}"'.format(self.cfg.port, self.cfg.port)
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

    def plot(self, plot_key: str, split_name: str, title_name: str, x: int, y: int, x_label: str = None, y_label: str = None):
        x_label = x_label or "Epochs"
        y_label = y_label or split_name
        if plot_key not in self.plots:
            self.plots[plot_key] = self.vis.line(
                X=np.array([x, x]),
                Y=np.array([y, y]),
                env=self.env,
                opts=dict(
                    legend=[split_name],
                    title=title_name,
                    xlabel=x_label,
                    ylabel=y_label,
                ),
            )
        else:
            self.vis.line(
                X=np.array([x]),
                Y=np.array([y]),
                env=self.env,
                win=self.plots[plot_key],
                name=split_name,
                update="append",
            )


if __name__ == "__main__":
    make_visdom()
