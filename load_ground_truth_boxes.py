import os
import pickle
import numpy as np
import time as time
import pandas as pd
from xml.dom import minidom
import holoviews as hv
import utilsss as ut
hv.extension('matplotlib')

###########################
### using the xml files ###
###########################

anotations = minidom.parse('/media/pedro/DATAPART1/AGOnIA/datasets_figure/notazione finale/NEU LARGE/neurofinder.01.01.xml')

neurons = anotations.getElementsByTagName('object')
print(len(neurons))
boxes = []
for i,n in enumerate(neurons):
    xmin = n.getElementsByTagName('bndbox')[0].getElementsByTagName('xmin')[0].childNodes[0].data
    ymin = n.getElementsByTagName('bndbox')[0].getElementsByTagName('ymin')[0].childNodes[0].data
    xmax = n.getElementsByTagName('bndbox')[0].getElementsByTagName('xmax')[0].childNodes[0].data
    ymax = n.getElementsByTagName('bndbox')[0].getElementsByTagName('ymax')[0].childNodes[0].data
    boxes.append([int(xmin),int(ymin),int(xmax),int(ymax)])
np.array(boxes).shape

###########################
### using the csv files ###
###########################

anotations_all = pd.read_csv('/media/pedro/DATAPART1/AGOnIA/datasets_figure/notazione finale/NEU LARGE/test.csv')

anotations_all.columns=['experiment','xmin','ymin','xmax','ymax','id']
anotations_all.loc[anotations_all['experiment']=='neurofinder.01.01.bmp']
boxes_csv = []
for index, row in anotations_all.loc[anotations_all['experiment']=='neurofinder.01.01.bmp'].iterrows():
    boxes_csv.append([row['xmin'],row['ymin'],row['xmax'],row['ymax']])
np.shape(boxes_csv)
np.shape(boxes)

##################################
### load agonia boxes and plot ###
##################################


data_path = '/media/pedro/DATAPART1/AGOnIA/datasets_figure/neurofinder/neurofinder.01.01'
data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)

boxes_agonia = pickle.load(open(boxes_path,'rb'))
boxes_agonia=boxes_agonia[boxes_agonia[:,4]>.15].astype(int)[:,:4]
boxes_agonia.shape
roi_bounds = hv.Path([hv.Bounds(tuple([roi[0],median_projection.shape[0]-roi[1],roi[2],median_projection.shape[0]-roi[3]])) for roi in boxes_agonia[:,:4]]).options(color='red')
roi_bounds_true = hv.Path([hv.Bounds(tuple([roi[0],median_projection.shape[0]-roi[1],roi[2],median_projection.shape[0]-roi[3]])) for roi in boxes_csv]).options(color='green')
img = hv.Image(median_projection,bounds=(0,0,median_projection.shape[1],median_projection.shape[0])).options(cmap='gray')
(img*roi_bounds*roi_bounds_true).opts(fig_size=300)
