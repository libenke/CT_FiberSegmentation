#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy import ndimage as ndi
import pandas as pd
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize_3d
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops_table
from skimage.exposure import equalize_adapthist
import imageio.v2 as imageio
import glob
import os


# # Load Data
####################################################################
# fill the datas
#dir_name is the director to store all the tmp and final results 
data_dirs = ["demo_data"]
####################################################################
for data_dir in data_dirs:
    #The inital index of data is z,y,x
    imagelist = [imageio.imread(file) for file in glob.glob(os.path.join(data_dir,"*[0-9][0-9][0-9].tif*"))]
    data = np.array(imagelist)
    del(imagelist)
    print("initial data dimensions:", data.shape)
    #change the index of data to x,y,z, and change the matrix from view to continous stored array
    data = np.ascontiguousarray(np.moveaxis(data,source=[0,2],destination=[2,0]))
    print("tunned data dimensions:", data.shape)


    # # Threshold and Skeletonization
    #binarization
    data = equalize_adapthist(data)
    thresholds = threshold_multiotsu(data)
    mask = ndi.binary_fill_holes((np.digitize(data, thresholds) == 2)).astype(np.uint8)
    mask = ndi.binary_opening(input=mask,structure=np.ones([3,3,3],dtype=int))
    #skeletonization
    skeleton = skeletonize_3d(mask)
    skeleton[skeleton.nonzero()]=1

    # # Cut the branches of the skeleton according to the orientation field calculated in previous step
    #culculate the neibours of each points
    from skimage.morphology import cube
    neigbours = ndi.convolve(skeleton,cube(3),mode='constant',cval=0)
    print("cross point amounts: ",np.sum((neigbours>=4) & skeleton))

    #load the orientation field, calculated by step1
    px = np.load(os.path.join(data_dir,"px.npy"))
    py = np.load(os.path.join(data_dir,"py.npy"))
    pz = np.load(os.path.join(data_dir,"pz.npy"))
    correlation = np.load(os.path.join(data_dir,"correlation.npy"))

    #cut the branches based on orientation field
    for points_index, (x,y,z) in enumerate(zip(*np.where((neigbours>=4) & skeleton))):
        if points_index%10000 == 0:
            print("Already treat points amouts:", points_index)
        #remove the cross points in the boundery
        if (x <= 1) | (y <= 1) | (z <= 1) | (x >= skeleton.shape[0]-2) | (y >= skeleton.shape[1]-2) | (z >= skeleton.shape[2]-2):
            skeleton[x,y,z] == 0
            continue
        #recalculate the neighbors of this points
        cube3_ref = skeleton[x-1:x+2,y-1:y+2,z-1:z+2]# numpy is default pass by fererence
        cube3_copy = np.copy(cube3_ref)
        cube3_copy[1,1,1] = 0
        connectivity = np.sum(cube3_copy)
        if connectivity <= 2:
            continue
        #store all the branches information in the pandas table
        branches_df = pd.DataFrame(cube3_copy.nonzero(),index=["bx","by","bz"]).T
        #read the orientation information from the orientation field
        orient_vec = np.array([px[x,y,z],py[x,y,z],pz[x,y,z]])
        #calculate the correlations of all the branches with the orientation field
        for ind, row in branches_df.iterrows():
            bx = int(row["bx"])
            by = int(row["by"])
            bz = int(row["bz"])

            cube3_branch = np.copy(skeleton[x+bx-2:x+bx+1,y+by-2:y+by+1,z+bz-2:z+bz+1])
            cube3_branch[1,1,1] = 0
            cube3_branch[-bx,-by,-bz] = 0
            #find if any mother branches point in this cube, if any, set it to zero
            tmp_df = branches_df.loc[:,["bx","by","bz"]]-[bx,by,bz]
            tmp_df = tmp_df[(tmp_df["bx"] >= -1) & (tmp_df["bx"] <= 1) & (tmp_df["by"] >= -1) & (tmp_df["by"] <= 1) & (tmp_df["bz"] >= -1) & (tmp_df["bz"] <= 1)]
            if len(tmp_df) > 1:
                cube3_branch[tmp_df["bx"]+1,tmp_df["by"]+1,tmp_df["bz"]+1] = 0

            #store all the points in this branch in this numpy array, rows are (x,y,z) respectively
            #the relative value to the center point of x,y,z
            branch_points = np.array(cube3_branch.nonzero())
            branch_points[0,:] += bx-2
            branch_points[1,:] += by-2
            branch_points[2,:] += bz-2
            branch_points = np.c_[branch_points,[bx-1,by-1,bz-1]]
            #calculate the orientation vector
            orient_vec_b = np.sum(branch_points,axis=1).astype(np.float32)
            orient_vec_b /= np.linalg.norm(orient_vec_b)
            #update the remained 2 branches
            branches_df.loc[ind,"correlation"] = np.abs(np.dot(orient_vec_b,orient_vec))
        #sort the results by correlations,
        branches_df = branches_df.sort_values(by = "correlation",ascending=False)
        #remove all the branches with lower correlations, only kept the largest correlated two branches
        d_df = branches_df.iloc[2:]
        cube3_ref[d_df["bx"],d_df["by"],d_df["bz"]] = 0

    #check by re-culculating the neibours of each points
    from skimage.morphology import cube
    neigbours = ndi.convolve(skeleton,cube(3),mode='constant',cval=0)
    print("cross point amounts after tracing: ",np.sum((neigbours>=4) & skeleton))

    # # label and calculate properties
    #remove small objects, and label
    from skimage.morphology import remove_small_objects
    skeleton_final = remove_small_objects(skeleton.astype(bool),min_size=30,connectivity=3)
    label_skeleton = label(label_image=skeleton_final, connectivity=3)
    np.save(os.path.join(data_dir,"skeleton_final.npy"),skeleton_final)
    print("saved skeleton mask file to data folder")
    
    #calculate properties using pandas package
    skeleton_df = pd.DataFrame(np.array(label_skeleton.nonzero()).T,columns=["x","y","z"])
    skeleton_df["label"] = label_skeleton[skeleton_df["x"],skeleton_df["y"],skeleton_df["z"]]
    regions_df = pd.DataFrame(regionprops_table(label_skeleton,properties=('label', 'area',"centroid")))
    regions_df = regions_df.set_index("label")
    #skeleton_df = pd.merge(left=skeleton_df,right=regions_df,how="inner",on=["label"])
    #calculate orientation of individual fiber by cov ops 
    cov_df = skeleton_df.groupby(by = "label").cov()
    for label_ind, row in regions_df.iterrows():
        cov_mat = cov_df.loc[label_ind,:]
        eig_value, eig_mat = np.linalg.eig(cov_mat)
        eig_vec = eig_mat[:,np.argmax(eig_value)]
        regions_df.loc[label_ind,"px"] = eig_vec[0]
        regions_df.loc[label_ind,"py"] = eig_vec[1]
        regions_df.loc[label_ind,"pz"] = eig_vec[2]
    #calculate the orientation tensor of glass fibers
    regions_df["Axx"] = regions_df["px"]**2
    regions_df["Axy"] = regions_df["px"]*regions_df["py"]
    regions_df["Axz"] = regions_df["px"]*regions_df["pz"]
    regions_df["Ayy"] = regions_df["py"]**2
    regions_df["Ayz"] = regions_df["py"]*regions_df["pz"]
    regions_df["Azz"] = regions_df["pz"]**2

    #print average orientation tensor A
    print("The Orientation Tensor is:")
    [[regions_df.Axx.mean(), regions_df.Axy.mean(), regions_df.Axz.mean()], \
    [regions_df.Axy.mean(), regions_df.Ayy.mean(), regions_df.Ayz.mean(),], \
    [regions_df.Axz.mean(), regions_df.Ayz.mean(), regions_df.Azz.mean()]]

    #slice layers along y axis
    y_inds = regions_df["centroid-1"]
    layer_thick = (y_inds.max()-y_inds.min())/16
    regions_df["layer"] = np.rint((y_inds/layer_thick)).astype(int)
    #average values
    layer_df = regions_df.groupby(by = "layer").mean().loc[:,["Axx","Axy","Axz","Ayy","Ayz","Azz"]]
    #plot
    #layer_df.plot()

    #save data
    layer_df.to_csv(os.path.join(data_dir,"layer.csv"))
    with pd.HDFStore(os.path.join(data_dir,"DataFrame.h5"),mode="a") as store:
        store["regions_df"] = regions_df
        store["layer_df"] = layer_df
        store["skeleton_df"] = skeleton_df
    print("saved dataframe files to data folder")
