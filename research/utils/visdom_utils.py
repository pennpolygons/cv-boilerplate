import os
import time
import hydra
import matplotlib

import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, List
from dataclasses import dataclass, field
from omegaconf import DictConfig
from visdom import server, Visdom


#matplotlib.use("tkagg")


@hydra.main(config_path="../configs", config_name="default.yaml")
def make_visdom(cfg: DictConfig):
    vdb = Visualizer(cfg)


@dataclass
class VisPlot:
    ######## Required for Visdom line plot ########

    var_name: str  # Field name in the engine state (i.e. engine.outputs.var_name)
    plot_key: str  # Plot used to plot data. Non-existant plot key creates new plot
    split: str  # Legend split

    ################################################
    env: str = None  # leads to Visdom default "main" env
    opts: field(default_factory=dict) = None


@dataclass
class VisImg:
    ######## Required for Visdom image ########

    var_name: str
    img_key: str
    ################################################
    env: str = None  # leads to Visdom default "main" env
    opts: field(default_factory=dict) = None


class Visualizer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg.visdom

        visdom_command = 'screen -S visdom_{} -d -m bash -c "python -m visdom.server -port {}"'.format(
            self.cfg.port, self.cfg.port
        )

        os.mkdir("visdom")
        os.system(visdom_command)
        time.sleep(2)
        # self.env = self.cfg.default_env_name  # TODO: What is this
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

    def img_result(
        self,
        img_list: List[np.ndarray],
        nrow: int,
        caption: str = "view",
        title: str = "title",
        win: int = 1,
        env: str = None,
    ):
        self.vis.images(
            img_list,
            nrow=nrow,
            win=win,
            opts={"caption": caption, "title": title},
            env=env,
        )

    def plot_img_255(
        self, img_key: str, img: np.ndarray, env: str = None, opts: Dict = None,
    ):
        """Visdom plot a single image (channels-first CxHxW)"""
        self.vis.image(img, win=img_key, opts=opts, env=env)

    def plot_matplotlib(self, fig, caption="view", title="title", win=1, env=None):
        self.vis.matplot(
            fig,
            win=win,
            opts={"caption": caption, "title": title, "resizable": True},
            env=env,
        )

    def plot_plotly(self, fig, caption="view", title="title", win=1, env=None):
        self.vis.plotlyplot(fig, win=win, env=env)

    def plot(
        self,
        plot_key: str,
        split_name: str,
        x: int,
        y: int,
        env: str = None,
        opts: Dict = None,
    ):
        """Visdom plot line data"""
        if plot_key not in self.plots:
            self.plots[plot_key] = self.vis.line(
                X=np.array([x, x]),
                Y=np.array([y, y]),
                env=env,
                opts={**opts, "legend": [split_name]},
            )
        else:
            self.vis.line(
                X=np.array([x]),
                Y=np.array([y]),
                env=env,
                win=self.plots[plot_key],
                name=split_name,
                update="append",
            )

"""
    def plot_line(
        self,
        plot_key: str,
        split_name: str,
        x: int,
        y: int,
        env: str = None,
        opts: Dict = None,
    ):
        """Visdom plot line data. e denotes the option to have a new 
        window for every time you call plot_line"""

        if plot_key == "e":
            plot_key = "e" + str(self.counter)
            self.counter = self.counter + 1

        if plot_key not in self.plots:

            if y.shape[0] == 2:
                for i in range(y.shape[0]):
                    if i == 0:
                        self.plots[plot_key] = self.vis.line(
                            X=x, Y=y[i], env=env, opts={**opts},
                        )
                    else:
                        self.vis.line(
                            X=x,
                            Y=y[i],
                            env=env,
                            name=split_name,
                            update="append",
                            win=self.plots[plot_key],
                        )

            elif y.shape[0] == 1:
                self.plots[plot_key] = self.vis.line(
                    X=x, Y=y, env=env, opts={**opts, "legend": [split_name]},
                )
            else:
                print(
                    "Error: you are not supposed to have more than 3 dimensions here."
                )

        else:

            if y.shape[0] == 2:
                for i in range(y.shape[0]):
                    if i == 0:
                        win = self.vis.line(
                            X=x, Y=y[i], env=env, opts={**opts, "legend": [split_name]},
                        )
                    else:
                        self.vis.line(
                            X=x,
                            Y=y[i],
                            env=env,
                            name=split_name,
                            update="append",
                            win=win,
                        )
            elif y.shape[0] == 1:
                win = self.vis.line(
                    X=x, Y=y, env=env, opts={**opts, "legend": [split_name]},
                )
            else:
                print(
                    "Error: you are not supposed to have more than 3 dimensions here."
                )"""

if __name__ == "__main__":
    make_visdom()
