# segmentation Mariangela's data
import os
import time
import pickle
import warnings
import scipy.stats
import numpy as np
import pandas as pd
import caiman as cm
import seaborn as sns
import holoviews as hv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import utils_mari as ut
hv.extension('matplotlib')


# Do the segmentation with agonia
data_path = '/media/pedro/DATAPART1/Mariangela/20200313/area02_0002-1561'       #folder where the tif movie and the median projection are
data_name,median_projection,boxes_path = ut.get_files_names(data_path)
ut.agonia_detect(data_path,data_name,median_projection,multiplier=.7)           #boxes are saved as pickle files


# extract and save traces
agonia_th = .2 #choose a threshold of confidence for the detected boxes
traces = ut.traces_extraction_AGONIA(data_path,agonia_th)
#save results
df = pd.DataFrame(traces)
df.to_hdf(os.path.join(data_path,os.path.splitext(data_name[0])[0] + '_traces.hdf5'),key='df',mode='w')
