import os
import cv2 as cv
import numpy as np
import pandas as pd
import caiman as cm
import utilsss as ut
from tifffile import imsave
import matplotlib.pyplot as plt

def memmap_movie(fnames,load=True):
    '''memmap already motion corrected movie using caiman functions.
       if load return the memmap shaped ready to fit caiman (using cnm.fit(images))'''
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=12, single_thread=False)

    fname_new = cm.save_memmap(fnames, base_name='memmap_', order='C',
                           border_to_0=0, dview=dview) # exclude borders
    cm.stop_server(dview=dview)
    if load:
        Yr, dims, T = cm.load_memmap(fname_new)
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        return images

def undistort(focal, src, gt=[]):
    width  = src.shape[1]
    height = src.shape[0]
    distCoeff = np.zeros((4,1),np.float64)
    k2 = -6.24e-4; # negative to remove barrel distortion
    k1= 1.95e-5;
    p1 = 0.0;
    p2 = 0.0;
    distCoeff[0,0] = k1;
    distCoeff[1,0] = k2;
    distCoeff[2,0] = p1;
    distCoeff[3,0] = p2;

    cam = np.eye(3,dtype=np.float32)

    cam[0,2] = width/2.0  # define center x
    cam[1,2] = height/2.0 # define center y
    cam[0,0] = focal        # define focal length x
    cam[1,1] = focal        # define focal length y

    # here the undistortion will be computed
    dst = cv.undistort(src,cam,distCoeff)
    if not len(gt):
        return dst
    else:
        min_points=gt[:,:2]
        max_points=gt[:,2:]
        dst_min_points=cv.undistortPoints(min_points.astype(np.float32),cam,distCoeff, np.eye(3,dtype=np.float32), cam).reshape(min_points.shape[0],2)
        dst_max_points=cv.undistortPoints(max_points.astype(np.float32),cam,distCoeff, np.eye(3,dtype=np.float32), cam).reshape(max_points.shape[0],2)
        dst_annotation=np.concatenate([dst_min_points, dst_max_points], axis=1)
        filtered_annotations=[]
        for b in dst_annotation:
            if b[0]>=0 and b[1]>=0 and b[2]<width and b[3]<height and b[2]>b[0] and b[3]>b[1]:
                filtered_annotations.append(b)
        filtered_annotations=np.array(filtered_annotations)
        return dst, filtered_annotations

def correct_movie(images,focal):
    corrected = np.empty(images.shape)
    for i,image in enumerate(images):
        dst = undistort(focal, image)
        corrected[i] = dst
    return corrected

data_path = '/media/pedro/DATAPART1/AGOnIA/VPM'
save_path = '/media/pedro/DATAPART1/AGOnIA/VPM_corrected'
luca_params = pd.read_csv('/media/pedro/DATAPART1/AGOnIA/params/VPM_log (copy).txt',sep=" ")
luca_params.set_index('name',inplace=True)

for folder in next(os.walk(data_path))[1]:
    print(folder)
    focal = luca_params.loc[folder+'.bmp']['focal']
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(os.path.join(data_path,folder))
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    corrected = correct_movie(images,focal)
    imsave(os.path.join(save_path,folder,folder+'_corrected.tif'), corrected.astype('float32'))


#
