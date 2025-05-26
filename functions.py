#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 17:58:48 2023

@author: hannamanko
"""


import numpy as np
import cupy as cp
from scipy import fftpack
from scipy.fftpack import fft, ifft, hilbert
from PIL import Image
import matplotlib.pyplot as plt
from numpy import where
import math
from tifffile import imread, imsave
from skimage import io, measure
import sys
import pymmcore


from scipy import fft
import os.path

import warnings
from astropy.modeling import models, fitting

#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('matplotlib', 'qt5')



######################################    Acquisitions
###################
###################
def norm_data(data): # normalize data to have mean=0 and standard_deviation=1
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data-mean_data)/(std_data)

def ncc(data0, data1): #    normalized cross-correlation coefficient between two data sets
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))

def norm_data_cupy(data): # normalize data to have mean=0 and standard_deviation=1
    mean_data=cp.mean(data)
    std_data=cp.std(data, ddof=1)
    #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data-mean_data)/(std_data)

def ncc_cupy(data0, data1): #    normalized cross-correlation coefficient between two data sets
    return (1.0/(data0.size-1)) * cp.sum(norm_data_cupy(cp.array(data0))*norm_data_cupy(cp.array(data1)))


def file_name_check(path):
    filename, extension = os.path.splitext(path)
    counter = 2
    while os.path.exists(path):
        path = filename + "_" +str(counter)+""+ extension
        counter += 1
    return path


def line_select_callback(eclick, erelease):
    global x1, x2, y1, y2
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print(x1,x2,y1,y2)
    ax[1].plot(np.arange(0,len(stack[:,int(x1):int(x2), int(y1):int(x2)])),
              np.mean(np.mean(stack[:,int(x1):int(x2), int(y1):int(x2)],axis=1), axis=1))

def finish(event):
    global x1, x2, y1, y2, coords
    if event.key == 'n':
        coords[:] = x1,x2,y1,y2
        print(coords)
    return coords

mmc = pymmcore.CMMCore()


import time
###################
###################
######################################    Acquisitions


def testProcess():
    for i in range(0,100,10):
        time.sleep(2)
        print("something is happening")


#parameters:
Gamma = 29.6e-6
d = 1.5e-3
p = 19.5e-6
alpha = Gamma/(4*math.pi*d)
theta = .99

def minmax_scaler(arr, *, vmin=0, vmax=1):
    arr_min, arr_max = arr.min(), arr.max()
    return ((arr - arr_min) / (arr_max - arr_min)) * (vmax - vmin) + vmin


def grad_diff(image, mask_x, mask_y,Ix_ref,Iy_ref):
    fourier_mesure = fft.fftshift(fft.fft2(image, norm="ortho")).astype(np.complex64)
    Ix = fft.ifft2(fft.fftshift((fourier_mesure) * mask_x))
    Iy = fft.ifft2(fft.fftshift((fourier_mesure) * mask_y)) 
    #print("fft2", time.time()-start)
    DW1 = alpha*np.angle(Ix*np.conj(Ix_ref))
    DW2 = alpha*np.angle(Iy*np.conj(Iy_ref))   
    DWx = np.cos(theta) * DW1 - np.sin(theta) * DW2
    DWy = np.sin(theta) * DW1 + np.cos(theta) * DW2
    DWy[where(DWy>(DWy.mean()+DWy.std()*4))]=0
    DWy[where(DWy<(DWy.mean()-DWy.std()*4))]=0
    DWx[where(DWx>(DWx.mean()+DWx.std()*4))]=0
    DWx[where(DWx<(DWx.mean()-DWx.std()*4))]=0
    return DWx-DWy

def grad_diff_cupy(image, mask_x, mask_y,Ix_ref,Iy_ref):
    image = cp.array(image.astype(cp.complex64))
    fourier_mesure = cp.fft.fftshift(cp.fft.fft2(image, norm="ortho"))
    Ix = cp.fft.ifft2(cp.fft.fftshift((fourier_mesure) * cp.array(mask_x)))
    Iy = cp.fft.ifft2(cp.fft.fftshift((fourier_mesure) * cp.array(mask_y))) 
    DW1 = alpha*cp.angle(Ix*cp.conj(cp.array(Ix_ref)))
    DW2 = alpha*cp.angle(Iy*cp.conj(cp.array(Iy_ref)))   
    DWx = cp.cos(theta) * DW1 - cp.sin(theta) * DW2
    DWy = cp.sin(theta) * DW1 + cp.cos(theta) * DW2
    DWy[where(DWy>(DWy.mean()+DWy.std()*4))]=0
    DWy[where(DWy<(DWy.mean()-DWy.std()*4))]=0
    DWx[where(DWx>(DWx.mean()+DWx.std()*4))]=0
    DWx[where(DWx<(DWx.mean()-DWx.std()*4))]=0
    return DWx-DWy


def phase_image(image, image_ref, choose_coord=False):
    if len(image.shape) <3:
        im = image
    else:
        im  = image[1]
    if choose_coord == True:
        plt.figure()
        plt.imshow(np.log10(np.abs(fft.fftshift(fft.fft2(ref, norm="ortho")))))
        plt.title("Please click on two first harmonics to get phase \n then pres Enter\n (LEFT mouse to add point, RIGHT to remove)")
        points = plt.ginput(4) 
        plt.close()
        center1_x,center1_y,center2_x,center2_y = points[0][0],points[0][1] ,points[1][0],points[1][1]  
        #center1_x = points[0][0] 
        #center1_y = points[0][1] 
        #center2_x = points[1][0]
        #center2_y = points[1][1] 
    else:
        center1_x,center1_y,center2_x,center2_y = 1005, 320, 1170, 1003 
    R = image_ref.shape[0] // 6
    fourier_mesure = fft.fftshift(fft.fft2(image, norm="ortho")).astype(np.complex64)
    fourier_ref = fft.fftshift(fft.fft2(image_ref, norm="ortho")).astype(np.complex64)
    #print("fourier mes ref", time.time()-start)
    x, y = np.meshgrid(np.arange(image_ref.shape[1]), np.arange(image_ref.shape[0]))
    mask_x = np.sqrt((x - center1_x)**2 + (y - center1_y)**2) <= R
    mask_y = np.sqrt((x - center2_x)**2 + (y - center2_y)**2) <= R
    selected_x = (fourier_mesure) * mask_x
    selected_y = (fourier_mesure) * mask_y
    #plt.imshow(np.abs(np.log10(selected_x[0])))
    selected_x_ref = (fourier_ref) * mask_x
    selected_y_ref = (fourier_ref) * mask_y
    Ix = fft.ifft2(fft.fftshift(selected_x))
    Iy = fft.ifft2(fft.fftshift(selected_y)) 
    Ix_ref = fft.ifft2(fftpack.fftshift(selected_x_ref))
    Iy_ref = fft.ifft2(fftpack.fftshift(selected_y_ref))    
    #print("fft2", time.time()-start)
    DW1 = alpha*np.angle(Ix*np.conj(Ix_ref))
    DW2 = alpha*np.angle(Iy*np.conj(Iy_ref))   
    DWx = np.cos(theta) * DW1 - np.sin(theta) * DW2
    DWy = np.sin(theta) * DW1 + np.cos(theta) * DW2
    DWx = DWx - np.nanmean(DWx)
    DWy = DWy - np.nanmean(DWy)
    #print("Dxy ", time.time()-start)
    Nx = image_ref.shape[1]
    Ny = image_ref.shape[0]
    [kx, ky] = np.meshgrid(np.array(list(range(1,Nx+1)),dtype='float16'),np.array(list(range(1,Ny+1)),dtype='float16'))
    kx = kx-Nx/2-1
    ky = ky-Ny/2-1
    kx[(where(kx ==0))]=np.nan#float('inf')
    ky[(where(ky ==0))]=np.nan#float('inf')
    W0_ = fft.ifftshift((fft.fftshift(fft.fft2(DWx, norm="ortho").astype(np.complex64))+ 1j*fft.fftshift(fft.fft2(DWy, norm="ortho").astype(np.complex64)))/(1j*2*math.pi*(kx/Nx + 1j*ky/Ny)))
    #print("W0-", time.time()-start)
    W0_ =np.nan_to_num(W0_)
    W0_[(where(np.abs(W0_)== float('inf')))] = 0
    W0 = fft.ifft2(W0_) 
    W = p*W0.imag*1e9
    return W

def gradients_c(image, image_ref):
    if len(image.shape) <3:
        im = image
    else:
        im  = image[1]
        center1_x,center1_y,center2_x,center2_y = 1005, 320, 1170, 1003 
    R = image_ref.shape[0] // 6
    fourier_mesure = fft.fftshift(fft.fft2(image, norm="ortho")).astype(np.complex64)
    fourier_ref = fft.fftshift(fft.fft2(image_ref, norm="ortho")).astype(np.complex64)
    #print("fourier mes ref", time.time()-start)
    x, y = np.meshgrid(np.arange(image_ref.shape[1]), np.arange(image_ref.shape[0]))
    mask_x = np.sqrt((x - center1_x)**2 + (y - center1_y)**2) <= R
    mask_y = np.sqrt((x - center2_x)**2 + (y - center2_y)**2) <= R
    selected_x = (fourier_mesure) * mask_x
    selected_y = (fourier_mesure) * mask_y
    #plt.imshow(np.abs(np.log10(selected_x[0])))
    selected_x_ref = (fourier_ref) * mask_x
    selected_y_ref = (fourier_ref) * mask_y
    Ix = fft.ifft2(fft.fftshift(selected_x))
    Iy = fft.ifft2(fft.fftshift(selected_y)) 
    Ix_ref = fft.ifft2(fftpack.fftshift(selected_x_ref))
    Iy_ref = fft.ifft2(fftpack.fftshift(selected_y_ref))    
    #print("fft2", time.time()-start)
    DW1 = alpha*np.angle(Ix*np.conj(Ix_ref))
    DW2 = alpha*np.angle(Iy*np.conj(Iy_ref))   
    DWx = np.cos(theta) * DW1 - np.sin(theta) * DW2
    DWy = np.sin(theta) * DW1 + np.cos(theta) * DW2
    #DWx = DWx - np.nanmean(DWx)
    #DWy = DWy - np.nanmean(DWy)
    return DWx-DWy


################

def flattening(image_to_flatten, deg): 
    image_to_flatten.astype(np.float16)## return model
    nx = image_to_flatten.shape[-2]
    ny = image_to_flatten.shape[-1]
    np.random.seed(0)
    y, x = np.mgrid[:nx, :ny]  
    p_init = models.Polynomial2D(degree=deg)
    fit_p = fitting.LevMarLSQFitter()    
    with warnings.catch_warnings():# Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        fit = fit_p(p_init, x, y, z=image_to_flatten) 
    # Plot the data with the best-fit model
    plt.figure(figsize=(8,2.5))
    plt.subplot(1,3,1)
    plt.imshow(image_to_flatten,interpolation='nearest')
    plt.title("Data")
    plt.subplot(1,3,2)
    plt.imshow(fit(x, y), interpolation='nearest')
    plt.title("Model")
    plt.subplot(1,3,3)
    plt.imshow(image_to_flatten - fit(x, y),  interpolation='nearest')
    plt.title("Flatten image")
    return fit(x,y)

###############
def intensity_image(image, image_ref, choose_coord=False):
##################    Intensity image    \\\\\\\\\\\\
    ############
    if len(image.shape) <3:
        im = image
    else:
        im  = image[1]
    if choose_coord == True:
        plt.figure()
        plt.imshow(np.log10(np.abs(fft.fftshift(fft.fft2(im, norm="ortho")))))
        plt.title("Please lick on 0th harmonic (the central spot) \n then pres Enter\n (LEFT mouse to add point, RIGHT to remove)")
        cen_points = plt.ginput(2)    
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        center_x = cen_points[0][0]#960 
        center_y = cen_points[0][1]#960 
    else:
        center_x = image_ref.shape[0]/2 
        center_y = image_ref.shape[1]/2
    R = image_ref.shape[0] // 6
    x, y = np.meshgrid(np.arange(image_ref.shape[1]), np.arange(image_ref.shape[0]))
    mask = np.sqrt((x - center_x)**2 + (y - center_y)**2) < R
    fourier_mesure = fft.fftshift(fft.fft2(image, norm="ortho"))
    fourier_ref = fft.fftshift(fft.fft2(image_ref, norm="ortho"))
    selected = fourier_mesure * mask
    selected_ref = fourier_ref * mask    
    T =  (fft.ifft2(fft.ifftshift(selected))).real/(fft.ifft2(fft.ifftshift(selected_ref))).real   
    #plt.imshow(T, cmap = 'gray'), plt.colorbar()
    return T
##########3