import cv2
import matplotlib.pyplot as plt

plt.switch_backend("agg")
import numpy as np
import os
import PIL
import torch
import torchvision
import typing
import unittest


from matplotlib.pyplot import imshow
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.cm import get_cmap
from PIL import Image
from torchvision import models, transforms
from torch.nn import functional as F
from torch import topk

from typing import Union

TENSOR_OR_ARRAY = Union[torch.Tensor, np.ndarray]


def resample_lanczos(im: PIL.Image, W: int, H: int) -> np.ndarray:
    """Resize the image correctly. Thanks to Jaakko Lehtinin for the tip."""
    new_size = (W, H)
    im = im.resize(new_size, Image.LANCZOS)

    return np.array(im)


def read_image_resize(current_img_path: str, resize=(0, 0)) -> np.ndarray:
    """Input: path, tuple of resize shape (optional). Default will not resize
    Output: numpy array of wxhxc, range of pixels from 0 to 1"""
    im = Image.open(current_img_path)

    if resize[0] != 0:
        img = resample_lanczos(im, resize[0], resize[1])
    else:
        img = np.array(im)

    img = np.float32(img) / 255
    return img


def normalize_for_imagenet(original_img: torch.Tensor) -> torch.Tensor:
    """This function takes a channel-last numpy array and preprocceses it 
    so that it is normalized to ImageNet statistics. It returns the
    normalized numpy array channel-first. 

    input: numpy  array of image, bxcxwxh
    Output: numpy array of  either bxcxwxh  
    """

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = original_img.clone()

    if len(original_img.shape) == 4:

        for i in range(3):
            preprocessed_img[:, i, :, :] = preprocessed_img[:, i, :, :] - means[i]
            preprocessed_img[:, i, :, :] = preprocessed_img[:, i, :, :] / stds[i]

    else:
        raise ValueError("Input image missing a dimension!")

    return preprocessed_img


def convert_whc_to_cwh(img: TENSOR_OR_ARRAY) -> TENSOR_OR_ARRAY:
    """Takes in either tensor or numpy, returns channel first of the same type as input (numpy/tensor)"""

    if torch.is_tensor(img):
        if len(img.shape) == 4:
            preprocessed_img = img.permute(0, 3, 1, 2)
        else:
            preprocessed_img = img.permute(2, 0, 1)
    else:
        if len(img.shape) == 4:
            preprocessed_img = np.transpose(img, (0, 3, 1, 2))
        else:
            preprocessed_img = np.transpose(img, (2, 0, 1))

    return preprocessed_img


def convert_cwh_to_whc(img: TENSOR_OR_ARRAY) -> TENSOR_OR_ARRAY:
    """Takes in either tensor or numpy, returns channel last of the same type as input (numpy/tensor)"""
    if torch.is_tensor(img):
        if len(img.shape) == 4:
            preprocessed_img = img.permute(0, 2, 3, 1)
        else:
            preprocessed_img = img.permute(1, 2, 0)
    else:
        if len(img.shape) == 4:
            preprocessed_img = np.transpose(img, (0, 2, 3, 1))
        else:
            preprocessed_img = np.transpose(img, (1, 2, 0))

    return preprocessed_img


def inverse_imagenet_preprocess(img: np.ndarray) -> np.ndarray:
    """This function takes a channel-first numpy array and inverse normalizes
    it back to its original statistics before processing.

    input: numpy  array of image, of cxwxh
    Output: numpy array of  cxwxh"""

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    if len(img.shape) == 4:
        for i in range(3):
            img[:, i, :, :] = img[i, :, :] * stds[i]
            img[:, i, :, :] = img[i, :, :] + means[i]
    else:
        raise ValueError("Input image missing a dimension!")

    return img


def inverse_mnist_preprocess(img: np.ndarray) -> np.ndarray:
    """This function takes a channel-first numpy array and inverse normalizes
    it back to its original statistics before processing.

    input: numpy  array of image, of cxwxh
    Output: numpy array of  cxwxh"""
    means = [0.1307]
    stds = [0.3081]

    if len(img.shape) == 4:
        img[:, 0, :, :] = img[0, :, :] * stds[0]
        img[:, 0, :, :] = img[0, :, :] + means[0]
    else:
        raise ValueError("Input image missing a dimension!")

    return img


def create_M_matrix(thedx=None, thedy=None, scale=None, cuda=True) -> torch.Tensor:
    """Creating the affine transformation matrix for function below
    Input: thedx = 1-d torch tensor
    thedy = 1-d torch tensor of length n 
    scale = 1-d torch tensor of length n
    cuda =  boolean
    returns: nx2x3 matrix """

    dxdy = torch.cat((thedx, thedy), dim=0).reshape((2, thedx.shape[0])).permute((1, 0))
    dxdy_unsqueezed = torch.unsqueeze(dxdy, dim=2)  # 1xnx2x1
    scale = torch.unsqueeze(scale, dim=1)  # 1xnx1

    if cuda:
        ones = torch.cat(
            thedx.shape[0] * [torch.unsqueeze(torch.eye(2), dim=0)], dim=0
        ).to(gpu)
    else:
        ones = torch.cat(thedx.shape[0] * [torch.unsqueeze(torch.eye(2), dim=0)], dim=0)

    """Multiplying scale by identity 2x2 into a nx2x2"""
    two_by_twos = scale[:, :, None].float() * ones

    """Putting translation and scale together"""
    affine_matrices = torch.cat((two_by_twos, dxdy_unsqueezed.float()), dim=2)

    return affine_matrices


def affine_transform(
    img: torch.Tensor,
    affine_matrices: torch.Tensor,
    scale=True,
    translate=True,
    rotate=True,
) -> torch.Tensor:
    """Input arguments:
    img: b x c x w x h tensor or c x w x h tensor
    affine_matrices: 2x3 tensor
    Returns: transformed_img: bxcxwxh"""

    if len(img.shape) == 3:
        img = torch.unsqueeze(img, dim=0)

    grid = F.affine_grid(affine_matrices, list(img.shape))
    transformed_image = F.grid_sample(img, grid, mode="bilinear")

    return transformed_image


#################################################################################
###                        Functions for low-pass filtering
#################################################################################


def make_gaussian_filter(
    num_rows: int, num_cols: int, sigma: torch.Tensor, cuda: bool, high_pass: bool
) -> torch.Tensor:
    """Inputs:
    num_rows: scalar: the height of the image, now the size of the y dimension of the gaussian filter
    num_cols: scalar: the widht of the image, now the size of the x dimension of the gaussian filter 
    sigma: tensor of shape (n) the standard deviations to be used for both dimensions
    cuda:  boolean
    high_pass: boolean. If high_pass, invert the gaussian filter to do the opposite of low pass filter"""

    center_i = int(num_rows / 2) + 1 if num_rows % 2 == 1 else int(num_rows / 2)
    center_j = int(num_cols / 2) + 1 if num_cols % 2 == 1 else int(num_cols / 2)

    bb = torch.tensor(
        np.array(
            [
                [
                    -1.0 * ((i - center_i) ** 2 + (j - center_j) ** 2)
                    for j in range(num_cols)
                ]
                for i in range(num_rows)
            ]
        )
    )
    if cuda:
        cc_seven = (
            torch.cat(sigma.shape[0] * [torch.unsqueeze(bb, dim=0)], dim=0)
            .type(torch.FloatTensor)
            .cuda()
        )
        divide = torch.tensor(2 * sigma ** 2)[:, None, None].cuda()
    else:

        cc_seven = torch.cat(sigma.shape[0] * [torch.unsqueeze(bb, dim=0)], dim=0).type(
            torch.FloatTensor
        )
        divide = torch.tensor(2 * sigma ** 2)[:, None, None]

    vectorized = torch.exp(cc_seven / divide.float())

    return vectorized


def roll_n(X: torch.Tensor, axis: int, n: int) -> torch.Tensor:
    """Inputs:
    X: tensor of shape nxhxw, this is the n different real components of the fourier transforms
    of the image of size hxw
    axis: axis
    n: shift amount
    """

    f_idx = tuple(
        slice(None, None, None) if i != axis else slice(0, n, None)
        for i in range(X.dim())
    )
    b_idx = tuple(
        slice(None, None, None) if i != axis else slice(n, None, None)
        for i in range(X.dim())
    )
    front = X[f_idx]
    back = X[b_idx]

    return torch.cat([back, front], axis)


def batch_fftshift2d(x: torch.Tensor) -> torch.Tensor:
    """
    This functions centers the 2d fourier transform.
    Inputs:
    x: tensor of shape nxhxwx2, this is the n different fourier transforms of the image of size hxw
    Returns: shifted fft2d of nxhxwx2
    """

    real, imag = torch.unbind(x, -1)

    for dim in range(1, len(real.size())):
        n_shift = real.size(dim) // 2
        if real.size(dim) % 2 != 0:
            n_shift += 1
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)

    return torch.stack((real, imag), -1)


def batch_ifftshift2d(x: torch.Tensor) -> torch.Tensor:
    """
    This functions inverse centers the 2d fourier transform, aka brings it back to non-centered.
    Inputs:
    x: tensor of shape nxhxwx2, this is the n different center-shifted fourier transforms of the image of size hxw
    Returns: un-shifted fft2d of nxhxwx2
    """
    real, imag = torch.unbind(x, -1)

    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim) // 2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim) // 2)

    return torch.stack((real, imag), -1)


def filter_DFT(
    image_matrix: torch.Tensor, filter_matrix: torch.Tensor, cuda: bool
) -> torch.Tensor:
    """Inputs:
    image_matrix: tensor of shape hxw, 1d image 
    filter_matrix: tensor of shape nxhxw, this is n different hxw gaussian filters
    cuda: boolean
    Returns: tensor of shape nxhxw. This is the one channel image low-passed at those different thresholds"""

    if cuda:
        the_DFT = torch.rfft(
            image_matrix.type(torch.cuda.FloatTensor).cuda(), 2, onesided=False
        )

        real_and_img_fft = batch_fftshift2d(torch.tensor(the_DFT[None, :, :, :]))
        filtered_DFT_real = real_and_img_fft[:, :, :, 0] * filter_matrix.type(
            torch.cuda.FloatTensor
        )
        filtered_DFT_img = real_and_img_fft[:, :, :, 1] * filter_matrix.type(
            torch.cuda.FloatTensor
        )
    else:
        the_DFT = torch.rfft(image_matrix.type(torch.FloatTensor), 2, onesided=False)
        real_and_img_fft = batch_fftshift2d(torch.tensor(the_DFT[None, :, :, :]))
        filtered_DFT_real = real_and_img_fft[:, :, :, 0] * filter_matrix.type(
            torch.FloatTensor
        )
        filtered_DFT_img = real_and_img_fft[:, :, :, 1] * filter_matrix.type(
            torch.FloatTensor
        )

    total = batch_ifftshift2d(
        torch.cat(
            (filtered_DFT_real[:, :, :, None], filtered_DFT_img[:, :, :, None]), dim=3
        )
    )

    return torch.irfft(total, 2, onesided=False)


def low_pass(
    image_matrix: torch.Tensor, sigma: torch.Tensor, cuda: bool
) -> torch.Tensor:
    """Input:
    image_matrix : one channel image, tensor of shape hxw. 
    sigma: tensor of shape (n) of lowpass filtering thresholds
    cuda: boolean
    Returns: tensor of shape nxhxw. This is the one channel image low-passed at those different thresholds"""
    n, m = image_matrix.shape

    return filter_DFT(
        image_matrix, make_gaussian_filter(n, m, sigma, cuda, high_pass=False), cuda
    )


def low_passed_image(
    low_freq_img: torch.Tensor, sigma_low: torch.Tensor, cuda: bool
) -> torch.Tensor:
    """Inputs: 
    lowFreqImg: tensor of shape (cxwxh)
    sigmaLow: tensor of shape (n) of lowpass filtering thresholds
    cuda: boolean
    Returns: 
    new: tensor of shape (nxcxwxh). This is the image lowpassed at 
    the different thresholds """

    if cuda:
        new = torch.zeros(
            (sigma_low.shape[0], 3, low_freq_img.shape[1], low_freq_img.shape[2])
        ).cuda()
    else:
        new = torch.zeros(
            (sigma_low.shape[0], 3, low_freq_img.shape[1], low_freq_img.shape[2])
        )

    if cuda:
        sigma_low = sigma_low.cuda()

    low_passed_r = low_pass(low_freq_img[0, :, :], sigma_low, cuda)
    low_passed_g = low_pass(low_freq_img[1, :, :], sigma_low, cuda)
    low_passed_b = low_pass(low_freq_img[2, :, :], sigma_low, cuda)

    new[:, 0, :, :] = low_passed_r
    new[:, 1, :, :] = low_passed_g
    new[:, 2, :, :] = low_passed_b

    return new


if __name__ == "__main__":

    gpu = 0
    img_resized = read_image_resize("../dummy_data/dog_boat_bird.png", (224, 224))
    img_resized = torch.tensor(convert_whc_to_cwh(img_resized)).cuda()
    threshold = torch.tensor([10])
    low_img = low_passed_image(img_resized, threshold, True)

    """
    gpu = 0
    img_resized = read_image_resize("../dummy_data/dog_boat_bird.png", (224,224))
    img_resized = np.expand_dims(img_resized,axis=0)
    img_resized = torch.tensor(convert_whc_to_cwh(img_resized)).cuda()
    threshold = torch.tensor([10])
    low_img = low_passed_image(img_resized, threshold, True)
    img_resized_normal = normalize(img_resized)

    thedx = torch.tensor([1.0]).cuda()
    thedy = torch.tensor([1.0]).cuda()
    scale = torch.tensor([1.0]).cuda()

    the_matrix = create_M_matrix(thedx,thedy,scale)
    
    output = affine_transform(img_resized, the_matrix)
    
    current = convert_cwh_to_whc(output.detach().cpu().numpy())
    plt.imsave("lol.png",current[0])"""
