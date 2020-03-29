import os
os.chdir('/home/pedro/keras-retinanet')
import pickle
import numpy as np
import holoviews as hv
from skimage import io
from AGOnIA import AGOnIA
from holoviews import opts
from holoviews import streams
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from holoviews.streams import Stream, param

##########################################
########### necesary functions ###########
##########################################

def get_files_names(data_path):
    '''get paths of all the files needed for the analysis + median projection'''
    #initialize paths
    data_name = []
    median_projection = []
    boxes_path = []

    data_name = [data for data in os.listdir(data_path) if data.endswith('.tif')]
    try:
        median_path = os.path.join(data_path,[data for data in os.listdir(data_path) if data.endswith('.jpg')
                                          and data.startswith('MED')][0])
        if plt.imread(median_path).ndim==3:
            median_projection = (np.array(plt.imread(median_path),dtype='uint16')*255).mean(axis=2).astype('uint16')
        else:
            median_projection = np.array(plt.imread(median_path),dtype='uint16')*255
    except:
        print('Median projection is missing')

    # AGOnIA paths
    patches_path = [data for data in os.listdir(data_path) if data.endswith('boxes.pkl')]
    if patches_path:
        boxes_path = os.path.join(data_path,patches_path[0])

    return data_name,median_projection,boxes_path

def agonia_detect(data_path,data_name,median_projection,multiplier=1,th=0.05):
    '''detect cells with Agonia using the median projection and save results.
    Parameters
    ----------
    data_path : string
        path to folder containing the data
    data_name : list with string
        name of tiff series
    median_projection: ndarray_like
        output of get_files_names function. Is the median proyection of the
        t-series over which the detection is performed
    Multiplier: float
        only parameter needed for agonia, predefined for each dataset
    th: float
        threshold for detection confidence for each box, default is 0.05.
        It is recommended to detect using the default which is the minimun and
        filter afterwards'''
    det = AGOnIA('L4+CA1_800.h5',True)
    ROIs = det.detect(median_projection,threshold=th,multiplier=multiplier)
    pickle.dump( ROIs, open( os.path.join(data_path,os.path.splitext(data_name[0])[0] + '_boxes.pkl'), "wb" ) )


def _p_extract(patch,mima):
    mi,ma=np.percentile(patch,mima)
    sel=patch[patch>=mi]
    if ma>mi:
        sel=sel[sel<=ma]
    return np.mean(sel)


class Extractor:
    '''class for extracting boxes traces as implemented by AGOnIA'''
    def __init__(self, pmin=80, pmax=95, workers=12):
        from multiprocessing import Pool
        self.data=np.full((1,1),np.nan,np.double)
        self.mima=(pmin,pmax)
        self.pool=Pool(workers)

    def extract(self, frame, boxes):
        new_frame=np.full((1,self.data.shape[1]), np.nan, np.double)
        self.data=np.concatenate((self.data,new_frame),axis=0)
        if len(boxes):
            if len(boxes)-self.data.shape[1]+1>0:
                new_cells=np.full((self.data.shape[0],len(boxes)-self.data.shape[1]+1), np.nan, np.double)
                self.data=np.concatenate((self.data, new_cells), axis=1)
            patches=[]
            bg_frame=frame.astype(np.double, copy=True)
            for b in boxes:
                bg_frame[int(b[1]):int(b[3]+1),int(b[0]):int(b[2]+1)]=np.nan
                patches.append(frame[int(b[1]):int(b[3]+1),int(b[0]):int(b[2]+1)].flatten())
            traces=self.pool.starmap(_p_extract,[(p,self.mima) for p in patches])
            self.data[-1,1:]=traces
            self.data[-1,1:]-=np.nanmean(bg_frame)

    def get_traces(self, tail=0):
        if tail>0 and tail>self.data.shape[0]:
            return self.data[int(-tail):,1:]
        else:
            return self.data[1:,1:]

def traces_extraction_AGONIA(data_path,agonia_th):
    '''Extract traces of boxes using agonia Extractor class'''
    data_name,median_projection,boxes_path = get_files_names(data_path)
    images = io.imread(os.path.join(data_path,data_name[0]))

    with open(boxes_path,'rb') as f:
        boxes = pickle.load(f)
        f.close()
    # keep only cells above confidence threshold
    boxes.shape = boxes[boxes[:,4]>agonia_th].astype('int')

    ola = Extractor()
    for frame in images:
        ola.extract(frame,boxes)
    traces = ola.get_traces()
    return traces.T

def plot_AGonia_boxes(data_path,Score,box_idx):
    '''Function used as input for the dynamic map in function boxes_exploration():
    holoviews based plot to see Agonia boxes on top of median projection and mean trace
    for selected boxes.

    Parameters
    ----------
    data_path : string
        path to folder containing the data
    Score : float
        threshold for detection confidence for each box
    box_idx : int
        index of box to plot trace
    Returns
    -------
    Holoviews image to feed to dynamic map, selected cell is indicated with green square'''

    data_name,median_projection,boxes_path = get_files_names(data_path)
    #Yr, dims, T = cm.load_memmap(fname_new)
    #images = np.reshape(Yr.T, [T] + list(dims), order='F')
    images = io.imread(os.path.join(data_path,data_name[0]))
    with open(boxes_path,'rb') as f:
        boxes = pickle.load(f)
        f.close()
    boxes = boxes[boxes[:,4]>Score].astype('int')
    roi_bounds = hv.Path([hv.Bounds(tuple([roi[0],median_projection.shape[0]-roi[1],roi[2],median_projection.shape[0]
                                           -roi[3]])) for roi in boxes[:,:4]]).options(color='red')
    img = hv.Image(median_projection,bounds=(0,0,median_projection.shape[1],median_projection.shape[0]
                                                          )).options(cmap='gray')

    #box_trace = images[:,boxes[box_idx,1]:boxes[box_idx,3],boxes[box_idx,0]:boxes[box_idx,2]].mean(axis=(1,2))
    ola = Extractor()
    for frame in images:
        ola.extract(frame,[boxes[box_idx]])
    box_trace = ola.get_traces().squeeze()
    box_square = hv.Path([hv.Bounds(tuple([boxes[box_idx,0],median_projection.shape[0]-boxes[box_idx,1],boxes[box_idx,2],
                            median_projection.shape[0]-boxes[box_idx,3]]))]).options(color='lime')
    return ((img*roi_bounds*box_square).opts(width=600,height=600)+hv.Curve((np.linspace(0,len(box_trace)-1,
            len(box_trace)),box_trace),'Frame','Mean box Fluorescence').opts(width=600,framewise=True)).cols(1)


def boxes_exploration(data_path):
    '''Returns an interactive plot with the agonia boxes with a confidence value above
    a Score selected by the user with a slider. The user can see the mean-trace of a
    specific box and scroll through the boxes with a slider'''
    data_name,median_projection,boxes_path = get_files_names(data_path)
    with open(boxes_path,'rb') as f:
        boxes = pickle.load(f)
        f.close()

    kdims=[hv.Dimension('Score', values=np.arange(0.05,1,0.05)),
        hv.Dimension('box_idx',values=np.arange(0,len(boxes),1))]
    Experiment = Stream.define('Experiment', data_path=data_path)
    dmap = hv.DynamicMap(plot_AGonia_boxes, kdims=kdims,streams=[Experiment()])
    return dmap
