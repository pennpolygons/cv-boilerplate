import hydra
import os
import torchvision

from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.utils.data as utils_data
from typing import Tuple
from torchvision.transforms import Compose, ToTensor, Normalize
import pandas as pd
import numpy as np


def get_dataloaders(
    cfg: DictConfig, num_workers: int = 4, dataset_name: str = "mnist"
) -> Tuple[DataLoader, DataLoader]:
    """Return training and validation dataloaders"""

    ##################################################################
    # Change this. Demo dataset is MNIST
    ##################################################################
    if dataset_name == "mnist":
        data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

        dataset = torchvision.datasets.MNIST(
            os.path.join(hydra.utils.get_original_cwd(), cfg.dirs.data),
            download=True,
            transform=data_transform,
        )

        train_dataloader = DataLoader(
            dataset,
            batch_size=cfg.mode.train.batch_size,
            shuffle=cfg.mode.train.shuffle,
            num_workers=num_workers,
        )

        val_dataloader = DataLoader(
            dataset,
            batch_size=cfg.mode.val.batch_size,
            shuffle=cfg.mode.val.shuffle,
            num_workers=num_workers,
        )

    elif dataset_name == "reunion":

        (
            list_of_training_inputs,
            training_target_df,
            list_of_testing_inputs,
            testing_target_df,
        ) = process_uni_data(cfg)

        inputs_train = Variable(torch.FloatTensor(list_of_training_inputs))
        targets_train = Variable(torch.FloatTensor(training_target_df))
        inputs_test = Variable(torch.FloatTensor(list_of_testing_inputs))
        targets_test = Variable(torch.FloatTensor(testing_target_df))

        training_samples = utils_data.TensorDataset(inputs_train, targets_train)

        train_dataloader = utils_data.DataLoader(
            training_samples, batch_size=200, drop_last=False, shuffle=False
        )

        validation_samples = utils_data.TensorDataset(inputs_test, targets_test)

        val_dataloader = utils_data.DataLoader(
            validation_samples, batch_size=200, drop_last=False, shuffle=False
        )

    return train_dataloader, val_dataloader


def get_data_from_csv(cfg: DictConfig):

    list_of_dfs = []
    for csv in os.listdir(
        os.path.join(
            cfg.dirs.runtime_cwd,
            "../research/dataset/production_data/Consommation_universite-reunion",
        )
        # "research/dataset/production_data/Consommation_universite-reunion"
    ):

        df = pd.read_csv(
            os.path.join(
                cfg.dirs.runtime_cwd,
                "../research/dataset/production_data/Consommation_universite-reunion",
                csv,
            ),
            # "research/dataset/production_data/Consommation_universite-reunion/" + csv,
            engine="python",
            sep=";",
        )
        df.columns = [
            "datetime",
            "value",
            "cp",
            "c",
            "optimisation",
            "tarif",
            "unnamed",
        ]
        df.index = df.datetime
        df = df.set_index(pd.DatetimeIndex(df["datetime"]))
        list_of_dfs.append(df.value)

    finaldf = pd.concat(list_of_dfs)
    finaldf = finaldf.sort_index()
    finaldf = finaldf.apply(lambda x: int(x.split(",")[0]))
    df3 = finaldf.resample("60S").asfreq()
    all_data = df3["2016-01-01 00:00:00":"2017-12-31  00:00:00"][:-1]
    all_data = all_data.interpolate()
    all_data_ten = all_data[::10]
    # pdb.set_trace()
    all_data_ten_avg = np.mean(np.array(all_data).reshape(-1, 10), axis=1)
    all_data_ten = all_data_ten.interpolate()

    return all_data_ten, all_data_ten_avg


def process_uni_data(cfg: DictConfig):

    all_data_ten, all_data_ten_avg = get_data_from_csv(cfg)

    list_of_input = []
    list_of_output = []
    list_of_dates = []

    for i in range(int((len(all_data_ten)) / (6 * 24)) - 8):

        print("i", i)

        slices = []
        for j in range(7):  #

            curr_slice = all_data_ten_avg[
                i * 6 * 24 + j * 6 * 24 : i * 6 * 24 + 24 * 6 + j * 6 * 24
            ]
            slices.append(curr_slice)

        output = all_data_ten_avg[
            i * 6 * 24 + (7) * 24 * 6 : i * 24 * 6 + (7 + 1) * 24 * 6
        ]

        list_of_input.append(slices)
        list_of_output.append(np.array(output))
        list_of_dates.append(
            all_data_ten.index[17327 * 6 + (7) * 24 * 6 : 17327 * 6 + (7 + 1) * 24 * 6]
        )

    input_vals = np.array(list_of_input)

    target_vals = np.array(list_of_output)

    training_input_df = input_vals[:541] / 1777.0
    testing_input_df = input_vals[541:] / 1777.0
    training_target_df = target_vals[:541] / 1777.0
    testing_target_df = target_vals[541:] / 1777.0

    return (
        training_input_df.reshape(training_input_df.shape[0], -1),
        training_target_df,
        testing_input_df.reshape(testing_input_df.shape[0], -1),
        testing_target_df,
    )
