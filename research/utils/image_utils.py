import cv2
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
import pdb
import PIL
import plotly.graph_objects as go
import torch
import torchvision


from matplotlib.pyplot import imshow
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.cm import get_cmap
from PIL import Image
from plotly.subplots import make_subplots
from torchvision import models, transforms
from torch.nn import functional as F
from torch import topk



"""Read in image"""
def read_img(current_img_path):

    im = Image.open(current_img_path)

    return im

"""Resize the image correctly. Thanks to Jaakko Lehtinin for the tip."""
def resample_lanczos(im, W, H):

    '''Resize image to size (W, H).'''
    new_size = (W,H)
    im = im.resize(new_size, Image.LANCZOS)

    return np.array(im)

"""
This function takes a channel-last numpy array and preprocceses it 
so that it is normalized to ImageNet statistics. It returns the
normalized numpy array channel-first. 

input: numpy  array of image, either bxwxhxc or wxhxc
Output: numpy array of  either bxcxwxh or cxwxh"""
def normalize_for_imagenet(original_img):

    img = np.float32(original_img) / 255
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    preprocessed_img = img.copy()

    if  len(original_img.shape) == 4:
        for i in range(3):
            preprocessed_img[:,:, :, i] = preprocessed_img[:,:, :, i] - means[i]
            preprocessed_img[:,:, :, i] = preprocessed_img[:,:, :, i] / stds[i]    
        preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (0,3, 1, 2)))
    else:
        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]    
        preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    
    return preprocessed_img

"""Takes in either tensor or numpy, returns channel first"""
def convert_whc_to_cwh(img):

    if torch.is_tensor(img):
        if len(img.shape) == 4:
            preprocessed_img = img.permute(0,3,1,2)
        else:
            preprocessed_img = img.permute(2,0,1)
    else:
        if len(img.shape) == 4:
            preprocessed_img = np.transpose(img, (0,3,1,2))
        else:
            preprocessed_img = np.transpose(img, (2,0,1))
    
    return preprocessed_img

"""Takes in either tensor or numpy, returns channel last"""
def convert_cwh_to_whc(img):

    if torch.is_tensor(img):
        if len(img.shape) == 4:
            preprocessed_img = img.permute(0,2,3,1)
        else:
            preprocessed_img = img.permute(1,2,0)
    else:
        if len(img.shape) == 4:
            preprocessed_img = np.transpose(img, (0,2,3,1))
        else:
            preprocessed_img = np.transpose(img, (1,2,0))
    
    return preprocessed_img

"""
This function takes a channel-first numpy array and inverse normalizes
it back to its original statistics before processing.

input: numpy  array of image, of cxwxh
Output: numpy array of  cxwxh"""
def inverse_imagenet_preprocess(img):
    
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()
    
    for i in range(3):
        preprocessed_img[i, :, :] = preprocessed_img[i, :, :] * stds[i]
        preprocessed_img[i, :, :] = preprocessed_img[i, :, :] + means[i]
    
    return preprocessed_img

"""Creating the affine transformation matrix for function below
Input: thedx = 1-d torch tensor
thedy = 1-d torch tensor
scale = 1-d torch tensor
cuda =  boolean

returns: 2x3 matrix """
def create_M_matrix(thedx=None, thedy=None, scale=None, cuda=True):

    """Retrieving the translation parameters and putting them into a 1xnx2x1 tensor."""

    dxdy = torch.cat((thedx,thedy),dim=0).reshape((2,thedx.shape[0])).permute((1,0))
    dxdy_unsqueezed = torch.unsqueeze(dxdy,dim=2) #1xnx2x1
    scale = torch.unsqueeze(scale,dim=1) #1xnx1

    if cuda:
        ones = torch.cat(thedx.shape[0]*[torch.unsqueeze(torch.eye(2),dim=0)], dim=0).to(gpu)
    else:
        ones = torch.cat(thedx.shape[0]*[torch.unsqueeze(torch.eye(2),dim=0)], dim=0)

    """Multiplying scale by identity 2x2 into a nx2x2"""
    two_by_twos = scale[:,:,None].float() * ones

    """Putting translation and scale together"""
    affine_matrices = torch.cat((two_by_twos, dxdy_unsqueezed.float()), dim=2) 

    return affine_matrices

#def check_transformation(affine_matrices, scale=True, translate=True, rotate=True):

"""Input arguments:
img: b x c x w x h tensor or c x w x h tensor
    affine_matrices: 2x3 tensor
    Returns:
transformed_img: bxcxwxh"""

def affine_transform(img,affine_matrices, scale=True, translate=True, rotate=True):

    if len(img.shape)==3:
        img = torch.unsqueeze(img,dim=0)
    
    grid = F.affine_grid(affine_matrices,list(img.shape))    
    transformed_image = F.grid_sample(img, grid, mode='bilinear')

    return transformed_image

"""Functions for low-pass filtering"""

def scaleSpectrum(A):

    return np.real(np.log10(np.absolute(A) + np.ones(A.shape)))

def makeGaussianFilter(numRows, numCols, sigma, cuda, highPass):

    centerI = int(numRows/2) + 1 if numRows % 2 == 1 else int(numRows/2)
    centerJ = int(numCols/2) + 1 if numCols % 2 == 1 else int(numCols/2)

    bb = torch.tensor(np.array([[-1.0 * ((i - centerI)**2 + (j - centerJ)**2) for j in range(numCols)] for i in range(numRows)]))
    if cuda: 
        cc_seven = torch.cat(sigma.shape[0]*[torch.unsqueeze(bb,dim=0)], dim=0).type(torch.FloatTensor).cuda()
        divide = torch.tensor(2 * sigma**2)[:,None,None].cuda()
    else:
        
        cc_seven = torch.cat(sigma.shape[0]*[torch.unsqueeze(bb,dim=0)], dim=0).type(torch.FloatTensor)
        divide = torch.tensor(2 * sigma**2)[:,None,None]
    
    vectorized = torch.exp(cc_seven / divide)

    return vectorized

def roll_n(X, axis, n):

    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]

    return torch.cat([back, front], axis)

def batch_fftshift2d(x):

    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)

    return torch.stack((real, imag), -1) 

def batch_ifftshift2d(x):

    real, imag = torch.unbind(x, -1)

    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)

    return torch.stack((real, imag), -1)  
    
def filterDFT(imageMatrix, filterMatrix, cuda):
    
    if cuda:
        theDFT = torch.rfft(imageMatrix.type(torch.cuda.FloatTensor).cuda(),2, onesided=False)
        real_and_img_fft = batch_fftshift2d(torch.tensor(theDFT[None,:,:,:]))
        filteredDFT_real = real_and_img_fft[:,:,:,0]* filterMatrix.type(torch.cuda.FloatTensor)
        filteredDFT_img = real_and_img_fft[:,:,:,1]* filterMatrix.type(torch.cuda.FloatTensor)
    else:
        theDFT = torch.rfft(imageMatrix.type(torch.FloatTensor),2, onesided=False)
        real_and_img_fft = batch_fftshift2d(torch.tensor(theDFT[None,:,:,:]))
        filteredDFT_real = real_and_img_fft[:,:,:,0]* filterMatrix.type(torch.FloatTensor)
        filteredDFT_img = real_and_img_fft[:,:,:,1]* filterMatrix.type(torch.FloatTensor)
    total = batch_ifftshift2d(torch.cat((filteredDFT_real[:,:,:,None], filteredDFT_img[:,:,:,None]),dim=3))
    
    return torch.irfft(total,2,onesided=False) 

def lowPass(imageMatrix, sigma, cuda):

    n,m = imageMatrix.shape

    return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, cuda, highPass=False), cuda)
   
def lowPassedImage(lowFreqImg, sigmaLow,cuda):
    
    if cuda: 
        new = torch.zeros((sigmaLow.shape[0],3,lowFreqImg.shape[1],lowFreqImg.shape[2])).cuda()
    else:
        new = torch.zeros((sigmaLow.shape[0],3,lowFreqImg.shape[1],lowFreqImg.shape[2]))
    
    if cuda:
        sigmaLow = sigmaLow.cuda()
    
    lowPassed_r = lowPass(lowFreqImg[0,:,:], sigmaLow, cuda)
    lowPassed_g = lowPass(lowFreqImg[1,:,:], sigmaLow, cuda)
    lowPassed_b = lowPass(lowFreqImg[2,:,:], sigmaLow, cuda)
    
    new[:,0,:,:] = lowPassed_r
    new[:,1,:,:] = lowPassed_g
    new[:,2,:,:] = lowPassed_b

    return new

if __name__ == "__main__":

    gpu = 0
    img = read_img("../dummy_data/dog_boat_bird.png")
    resized_img = resample_lanczos(img,224,224)
    resized_img = np.float32(resized_img) / 255
    thedx = torch.tensor([1.0]).cuda()
    thedy = torch.tensor([1.0]).cuda()
    scale = torch.tensor([1.0]).cuda()
    the_matrix = create_M_matrix(thedx,thedy,scale)
    resized_img = convert_whc_to_cwh(resized_img)
    resized_img = torch.tensor(resized_img).cuda()
    output = affine_transform(resized_img, the_matrix)
    current = convert_cwh_to_whc(output.detach().cpu().numpy())
    plt.imsave("lol.png",current[0])


