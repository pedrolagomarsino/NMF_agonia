cd /home/pedro/keras-retinanet
from AGOnIA import AGOnIA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import pickle

#load the weights
det = AGOnIA('L4+CA1_800.h5',True)

#load data
median_path = '/home/pedro/Work/AGOnIA/Boxes-data/Seeded-Caiman/croped/MED_501271265_5000_crop.jpg'
max_path = '/home/pedro/Work/AGOnIA/Boxes-data/Seeded-Caiman/501271265_max.jpg'

median_projection = plt.imread(median_path)
median_projection = np.array(median_projection,dtype='uint16')*255 # uint16 goes to 255*255

max_projection = plt.imread(max_path)
max_projection = np.array(max_projection,dtype='uint8') # uint16 goes to 255*255

max_projection.dtype
max_projection.shape
# detect rois in median projection of the motion corrected movie
#ROIs = det.detect(median_projection,threshold=0.01,multiplier=2.1)
ROIs = det.detect(max_projection,threshold=0.01,multiplier=2)
ROIs.shape

# plot Boxes on top of median
fig,ax = plt.subplots(figsize=(24,12))
#ax.imshow(median_projection)
ax.imshow(max_projection)
for i in range(ROIs.shape[0]):
    rect = Rectangle((ROIs[i,0],ROIs[i,1]), ROIs[i,2]-ROIs[i,0],
          ROIs[i,3]-ROIs[i,1],color='r',fill=False)
    ax.add_patch(rect)
    ax.set_title('AGONIA over median',fontsize=15)


# detect rois using tessels
tessel_ROIs = det.tessel_detect(median_projection,threshold = .001, n_tess = 2, multiplier=1)
tessel_ROIs.shape

# plot Boxes on top of median
fig,ax = plt.subplots(figsize=(24,12))
ax.imshow(median_projection)
for i in range(tessel_ROIs.shape[0]):
    rect = Rectangle((tessel_ROIs[i,0],tessel_ROIs[i,1]), tessel_ROIs[i,2]-tessel_ROIs[i,0],
          tessel_ROIs[i,3]-tessel_ROIs[i,1],color='r',fill=False)
    ax.add_patch(rect)
    ax.set_title('AGONIA over median',fontsize=15)


# Load the full-size image and the corresponding boxes
full_median_path = '/home/pedro/Work/AGOnIA/Boxes-data/Seeded-Caiman/MED_501271265_5000.jpg'
full_med = plt.imread(full_median_path).astype('uint16')*255
data_path = '/home/pedro/Work/AGOnIA/Boxes-data/Seeded-Caiman'
with open(os.path.join(data_path,'501271265_boxes.pkl'),'rb') as f:
    cajas = pickle.load(f)
    f.close()
full_ROIs = np.empty(np.shape(np.array(cajas[:,:4]))).astype('int')
full_ROIs[:,[0,2]] = np.array(cajas[:,[0,2]].astype('int'))
full_ROIs[:,[1,3]] = np.array(cajas[:,[1,3]].astype('int'))


# plot the detected boxes in the croped imaged compare to the full image
fig,ax = plt.subplots(figsize=(24,12))
ax.imshow(full_med)
for i in range(ROIs.shape[0]):
    rect = Rectangle((ROIs[i,0]+10,ROIs[i,1]+10), ROIs[i,2]-ROIs[i,0],
          ROIs[i,3]-ROIs[i,1],color='r',fill=False)
    ax.add_patch(rect)
for i in range(full_ROIs.shape[0]):
    rect = Rectangle((full_ROIs[i,0],full_ROIs[i,1]), full_ROIs[i,2]-full_ROIs[i,0],
          full_ROIs[i,3]-full_ROIs[i,1],color='white',fill=False)
    ax.add_patch(rect)


# detect using the median of the full trace (that's 5000 frames) compare to the old patches
full_ROIs_new = det.detect(full_med,threshold=0.01,multiplier=1.5)
full_ROIs_new.shape
fig,ax = plt.subplots(figsize=(24,12))
ax.imshow(full_med)
for i in range(full_ROIs_new.shape[0]):
    rect = Rectangle((full_ROIs_new[i,0],full_ROIs_new[i,1]), full_ROIs_new[i,2]-full_ROIs_new[i,0],
          full_ROIs_new[i,3]-full_ROIs_new[i,1],color='r',fill=False)
    ax.add_patch(rect)
for i in range(full_ROIs.shape[0]):
    rect = Rectangle((full_ROIs[i,0],full_ROIs[i,1]), full_ROIs[i,2]-full_ROIs[i,0],
          full_ROIs[i,3]-full_ROIs[i,1],color='white',fill=False)
    ax.add_patch(rect)

# compare the old patches with the detected ones in the max figure
fig,ax = plt.subplots(figsize=(24,12))
ax.imshow(max_projection)
for i in range(ROIs.shape[0]):
    rect = Rectangle((ROIs[i,0],ROIs[i,1]), ROIs[i,2]-ROIs[i,0],
          ROIs[i,3]-ROIs[i,1],color='r',fill=False)
    ax.add_patch(rect)
for i in range(full_ROIs.shape[0]):
    rect = Rectangle((full_ROIs[i,0],full_ROIs[i,1]), full_ROIs[i,2]-full_ROIs[i,0],
          full_ROIs[i,3]-full_ROIs[i,1],color='white',fill=False)
    ax.add_patch(rect)

ROIs.shape
full_ROIs.shape
