#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy import ndimage as ndi
#could be accelarate by GPU, just replace the "import numpy" and "scipy" as "import cupy" as below
#import cupy as np
#from cupy import ndimage as ndi
from matplotlib import pyplot as plt
import imageio.v2 as imageio
import glob
import os


# # Load Data
####################################################################
# fill the datas
#dir_name is the director to store all the tmp and final results 
data_dirs = ["11_cropped","22_cropped","80_cropped"]
####################################################################

for data_dir in data_dirs:
    #The inital index of data is z,y,x
    imagelist = [imageio.imread(file) for file in glob.glob(os.path.join(data_dir,"*[0-9][0-9][0-9].tiff"))]
    data = np.array(imagelist)
    del(imagelist)
    print("initial data dimensions:", data.shape)
    #change the index of data to x,y,z, and change the matrix from view to continous stored array
    data = np.ascontiguousarray(np.moveaxis(data,source=[0,2],destination=[2,0]))
    print("tunned data dimensions:", data.shape)


    # # generate orientional kernels for convolve operations
    # ref. https://en.wikipedia.org/wiki/Spherical_coordinate_system  
    # azimuthal angle $$\phi = tan^{-1}\left(\frac{y}{x}\right)$$
    # latidute (polar) angle $$\theta = sin^{-1}\left(\frac{\sqrt{x^2+y^2}}{|r|}\right)$$
    # Cartesian coordinates:
    # $$x=r\ sin\theta\ cos\phi$$
    # $$y=r\ sin\theta\ sin\phi$$
    # $$z=r\ cos\theta\$$
    # 
    # range:$$\phi \subseteq \left [ -\pi/2,\pi/2  \right ] \\
    # \theta \subseteq \left [ 0,\pi/2  \right ] $$

    # # Generate line kernels
    # 
    # Time efficient for cpu calculations of convolve operations
    def kernel_fun_2(theta, phi, fiber_length = 30):
        half_length = int(fiber_length/2)
        kernel = np.zeros([2*half_length+1,2*half_length+1,2*half_length+1],dtype=np.float32)
        r = np.linspace(-half_length,half_length,1000)
        px = np.rint(r * np.sin(theta) * np.cos(phi) + half_length).astype(int)
        py = np.rint(r * np.sin(theta) * np.sin(phi) + half_length).astype(int)
        pz = np.rint(r * np.cos(theta) + half_length).astype(int)
        kernel[px,py,pz] = 1
        kernel /= kernel.sum()
        return kernel
    ################################################################################
    #default parameters, length of the fiber in the kernel
    fiber_length = 30
    ################################################################################
    
    # # Calculate the orientations field of 3D volume image
    # Use line kernels rather than cylinder kernels to save time.
    # Could be accelate using cupy package (Nvidia GPU needed)
    px = np.zeros_like(data,dtype=np.float16)
    py = np.zeros_like(data,dtype=np.float16)
    pz = np.zeros_like(data,dtype=np.float16)
    correlation = np.zeros_like(data,dtype=np.uint16)

    phi_range = np.linspace(-np.pi/2,np.pi/2,36,dtype=np.float32)
    theta_range = np.linspace(0,np.pi/2,18,dtype=np.float32)
    import time
    st = time.time()
    for phi in phi_range:
        for theta in theta_range:
            sst = time.time()
            #kernel
            kernel = kernel_fun_2(theta = theta , phi = phi, fiber_length = fiber_length)
            #kernel should be flipped before convovle operation
            kernel = np.flip(kernel,(0,1,2))
            #
            correlation_tmp = ndi.convolve(data, kernel, mode = 'constant', cval = 0.0 )
            update_mask = correlation_tmp>correlation
            correlation[update_mask] = correlation_tmp[update_mask]
            px[update_mask] = np.sin(theta) * np.cos(phi)
            py[update_mask] = np.sin(theta) * np.sin(phi)
            pz[update_mask] = np.cos(theta)
            print("phi-->",phi," theta-->",theta," total_time: ", time.time()-st, "s"," step_time: ", time.time()-sst, "s")
    np.save(os.path.join(data_dir,"px.npy"),px)
    np.save(os.path.join(data_dir,"py.npy"),py)
    np.save(os.path.join(data_dir,"pz.npy"),pz)
    np.save(os.path.join(data_dir,"correlation.npy"),correlation)
