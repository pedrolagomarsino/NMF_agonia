import os
import sys
#os.chdir('/home/pedro/keras-retinanet')
os.chdir('/home/pedro/keras-retinanet/AGOnIA_release')
import pickle
import warnings
import numpy as np
import pandas as pd
import caiman as cm
import holoviews as hv
from skimage import io
#from AGOnIA import AGOnIA
import AGOnIA2 as ag
from holoviews import opts
from holoviews import streams
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from matplotlib.patches import Rectangle
from holoviews.streams import Stream, param
from caiman.motion_correction import MotionCorrect
from scipy.ndimage.measurements import center_of_mass
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.components_evaluation import compute_event_exceptionality
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning)
sys.path.insert(1, '/home/pedro/Work/Hippocampus/code')
import to_Pedro as sut

def NMF_dec(box, n_components=1,init=None,random_state=None):
    cell=boxes[box]
    ##Reshape data to have boxes as single columns x time
    X = np.array([cell[:,:,i].reshape((np.size(cell,0)*np.size(cell,1))) for i in range(np.size(cell,2))]).T
    # fit NMF
    model = NMF(n_components=n_components, init=init, random_state=random_state)
    W = model.fit_transform(X)
    H = model.components_
    return W,H

def get_files_names(data_path):
    '''get paths of all the files needed for the analysis + median projection'''
    #initialize paths
    data_name = []
    median_projection = []
    fnames = []
    fname_new = []
    results_caiman_path = []
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

    fnames = [os.path.join(data_path,data_name[0])]

    # Caiman paths
    mmap_path = [data for data in os.listdir(data_path) if data.endswith('.mmap') and data.startswith('memmap')]
    results_Caiman = [data for data in os.listdir(data_path) if data.endswith('results.hdf5')]
    if mmap_path:
        fname_new = os.path.join(data_path,mmap_path[0])
    if results_Caiman:
        results_caiman_path = os.path.join(data_path,results_Caiman[0])

    # AGOnIA paths
    patches_path = [data for data in os.listdir(data_path) if data.endswith('boxes.pkl')]
    if patches_path:
        boxes_path = os.path.join(data_path,patches_path[0])

    return data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path

def caiman_motion_correct(fnames,opts):
    '''do the motion correction of the Caiman pipeline and save results.
    Parameters
    ----------
    fnames : string
        name of tiff movie
    opts : caiman object with the options for motion correction'''

    #%% start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

    # first we create a motion correction object with the parameters specified
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True)
    border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0
    # memory map the file in order 'C'
    fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
    border_to_0=border_to_0, dview=dview) # exclude borders
    cm.stop_server(dview=dview)

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
    #det = AGOnIA('L4+CA1_800.h5',True)
    #ROIs = det.detect(median_projection,threshold=th,multiplier=multiplier)
    det = ag.Detector('/home/pedro/keras-retinanet/AGOnIA_release/7934_tf13',True)
    ROIs = det.detect(median_projection,threshold=th,multiplier=multiplier)
    pickle.dump( ROIs, open( os.path.join(data_path,os.path.splitext(data_name[0])[0] + '_boxes.pkl'), "wb" ) )

def seeded_Caiman_wAgonia(data_path,opts,agonia_th):
    '''Run Caiman using as seeds the boxes detected with Agonia and save results
    Parameters
    ----------
    data_path : string
        path to folder containing the data
    opts : caiman object with the options for motion correction
    agonia_th : float
        threshold for detection confidence for each box'''
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = get_files_names(data_path)

    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    #load frames in python format (T x X x Y)

    #%% restart cluster to clean up memory
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=12, single_thread=False)

    # Load the boxes in the pickle file into the ROIs arrays.
    with open(boxes_path,'rb') as f:
        cajas = pickle.load(f)
        f.close()
    cajas = cajas[cajas[:,4]>agonia_th]

    #use the average size of the Agonia boxes as the cell size parameter for Caiman
    half_width  = np.mean(cajas[:,2]-cajas[:,0])/2
    half_height = np.mean(cajas[:,3]-cajas[:,1])/2
    gSig= (half_width.astype(int),half_height.astype(int))
    opts_dict = {'gSig': gSig}
    opts.change_params(opts_dict);

    #Build masks with the Agonia boxes
    Ain = np.zeros((np.prod(images.shape[1:]),cajas.shape[0]),dtype=bool)
    for i,box in enumerate(cajas):
        frame = np.zeros(images.shape[1:])
        #note that the coordinates of the boxes are transpose with respect to Caiman objects
        frame[box[1].astype('int'):box[3].astype('int'),box[0].astype('int'):box[2].astype('int')]=1
        Ain[:,i] = frame.flatten('F')

    #Run analysis
    cnm_seeded = cnmf.CNMF(n_processes, params=opts, dview=dview, Ain=Ain)
    try:
        print('initiating fit')
        cnm_seeded.fit(images)
        print('saving results')
        #cnm_seeded.estimates.detrend_df_f(quantileMin=50,frames_window=750,detrend_only=False)
        cnm_seeded.save(os.path.join(data_path,os.path.splitext(data_name[0])[0] + '_analysis_results.hdf5'))
    except:
        print('Tenemos un problema...')

    cm.stop_server(dview=dview)

def trace_correlation(data_path, agonia_th, select_cells=False,plot_results=True,denoise=False):
    '''Calculate the correlation between the mean of the Agonia Box and the CaImAn
    factor.
    Parameters
    ----------
    data_path : string
        path to folder containing the data
    agonia_th : float
        threshold for detection confidence for each box
    select_cells: bool, optional
        if True get index of active cells
    plot_results: bool, optional
        if True do boxplot of correlation values for all cells, if selected_cells,
        use only active cells
    denoise : bool, optional
        if True subtract neuropil
    Returns
    -------
    cell_corr : numpy array
        corrcoef value for all cells
    idx_active : list
        if select_cells=True returns index of active cells, otherwise returns
        index of all cells
    boxes_traces : NxT ndarray
        temporal trace for the N box that has an asociated caiman factor'''

    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = get_files_names(data_path)
    # load Caiman results
    cnm = cnmf.load_CNMF(results_caiman_path)
    # calculate the centers of the CaImAn factors
    centers = np.empty((cnm.estimates.A.shape[1],2))
    for i,factor in enumerate(cnm.estimates.A.T):
        centers[i] = center_of_mass(factor.toarray().reshape(cnm.estimates.dims,order='F'))
    # load boxes
    with open(boxes_path,'rb') as f:
        boxes = pickle.load(f)
        f.close()
    # keep only cells above confidence threshold
    boxes = boxes[boxes[:,4]>agonia_th].astype('int')

    #delete boxes that do not have a caiman cell inside
    k = 0
    for cell,box in enumerate(boxes):
        idx_factor = [i for i,center in enumerate(centers) if center[0]>box[1] and
         center[0]<box[3] and center[1]>box[0] and center[1]<box[2]]
        if not idx_factor:
            boxes = np.delete(boxes,cell-k,axis=0)
            k += 1

    # Load video as 3D tensor (each plane is a frame)
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    boxes_traces = np.empty((boxes.shape[0],images.shape[0]))
    #calculate correlations between boxes and CaImAn factors
    cell_corr = np.empty(len(boxes_traces))
    neuropil_trace = np.zeros(T)
    if denoise:
        _,neuropil_trace,neuropil_power = substract_neuropil(data_path,agonia_th,100,80)
    for cell,box in enumerate(boxes):
        # calculate boxes traces as means over images
        #boxes_traces[cell] = images[:,box[1]:box[3],box[0]:box[2]].mean(axis=(1,2))-neuropil_trace

        boxes_traces[cell] = images[:,box[1]:box[3],box[0]:box[2]].mean(axis=(1,2))

        #for using the percentile criteria
        med = np.median(images[:,box[1]:box[3],box[0]:box[2]],axis=0)
        box_trace = images[:,box[1]:box[3],box[0]:box[2]]
        boxes_traces[cell] = box_trace[:,np.logical_and(med>np.percentile(med,80),med<np.percentile(med,95))].mean(axis=1)
        boxes_traces[cell] = boxes_traces[cell]-neuropil_trace*neuropil_power*.7
        #boxes_traces[cell] = boxes_traces[cell]-neuropil_trace*boxes_traces[cell].mean()
        # get the asociated CaImAn factor by checking if its center of mass is inside the box
        idx_factor = [i for i,center in enumerate(centers) if center[0]>box[1] and
         center[0]<box[3] and center[1]>box[0] and center[1]<box[2]]
        # in case there is more than one center inside the box choose the one closer to the center of the box
        if len(idx_factor)>1:
            idx_factor = [idx_factor[np.argmin([np.linalg.norm([(box[3]-box[1])/2,
                                    (box[2]-box[0])/2]-c) for c in centers[idx_factor]])]]
        cell_corr[cell] = np.corrcoef([cnm.estimates.C[idx_factor[0]],boxes_traces[cell]])[1,0]

    if select_cells:
        #select only active cells using CaImAn criteria
        fitness, _, _, _ = compute_event_exceptionality(boxes_traces)
        idx_active = [cell for cell,fit in enumerate(fitness) if fit<-20]
    else:
        idx_active = [cell for cell,_ in enumerate(boxes_traces)]

    if plot_results:
        corr_toplot = [corr for id,corr in enumerate(cell_corr) if id in idx_active and ~np.isnan(corr)]
        fig,ax = plt.subplots()
        p = ax.boxplot(corr_toplot)
        ax.set_ylim([-1,1])
        plt.text(1.2,0.8,'n_cells = {}'.format(len(corr_toplot)))

    return cell_corr, idx_active, boxes_traces

def run_caiman_pipeline(data_path,opts,refit=False,component_evaluation=False,fr=14.913,rf=15,decay_time=0.6):
    '''Run the caiman pipeline out of the box, without using Agonia seeds.

    Parameters
    ----------
    data_path : string
        path to folder containing the data
    opts : caiman object with option parameters
    refit : bool, optional
        if True re-run CNMF seeded on the selected components from the fitting
    component_evaluation : bool, optional
        if True filter components using shape and signal criteria
    fr : float
        frame rate of video
    rf : int
        half-size of the patches in pixels (in seeded is None)
    decay_time = float
        decay time of calcium indicator

    Returns
    -------
    cnm : out of the box results of caiman detection
    cnm2 : refited-filtered results if refit, component_evaluations flags are True'''

    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = get_files_names(data_path)
    opts_dict = {'only_init': True,
                 'rf': rf,#15
                 'fr': fr,#14.913,
                 'decay_time': decay_time}#0.6
    opts.change_params(opts_dict);

    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')

    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)

    if refit:
        cnm2 = cnm.refit(images, dview=dview)

    if component_evaluation:
        cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
        #cnm2.estimates.select_components(use_object=True)

    if 'dview' in locals():
        cm.stop_server(dview=dview)
    return cnm, cnm2

def plot_AGonia_boxes(data_path,Score,box_idx,traces=False):
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

    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = get_files_names(data_path)
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
    box_square = hv.Path([hv.Bounds(tuple([boxes[box_idx,0],median_projection.shape[0]-boxes[box_idx,1],boxes[box_idx,2],
    median_projection.shape[0]-boxes[box_idx,3]]))]).options(color='lime')
    if traces:
        ola = Extractor()
        for frame in images:
            ola.extract(frame,[boxes[box_idx]])
        box_trace,_ = ola.get_traces()
        box_trace = box_trace.squeeze()
        ola.pool.terminate()
        ret = ((img*roi_bounds*box_square).opts(width=600,height=600)+hv.Curve((np.linspace(0,len(box_trace)-1,
                len(box_trace)),box_trace),'Frame','Mean box Fluorescence').opts(width=600,framewise=True)).cols(1)
    else:
        ret = (img*roi_bounds*box_square).opts(width=600,height=600)
    return ret

def plot_AGonia_boxes_interactive(data_path,Score,x,y):
    '''Function used as input for the dynamic map in function boxes_exploration_interactive():
    Same as plot_AGonia_boxes but now instead of having as input the index of the box the user
    can click on a box and it will use the (x,y) coordinates of the click to find the related box.
    The output now is only the boxes (all in red, selected in green)'''
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = get_files_names(data_path)
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    with open(boxes_path,'rb') as f:
        boxes = pickle.load(f)
        f.close()
    boxes = boxes[boxes[:,4]>Score].astype('int')
    roi_bounds = hv.Path([hv.Bounds(tuple([roi[0],median_projection.shape[0]-roi[3],roi[2],median_projection.shape[0]
                                           -roi[1]])) for roi in boxes[:,:4]]).options(color='red')

    if None not in [x,y]:
        try:
            box_idx = [i for i,box in enumerate(boxes) if x<box[2] and x>box[0] and y<(median_projection.shape[0]-box[1])
                    and y>(median_projection.shape[0]-box[3])][0]
        except:
            pass
    else:
        box_idx = 0
    #box_trace = images[:,boxes[box_idx,1]:boxes[box_idx,3],boxes[box_idx,0]:boxes[box_idx,2]].mean(axis=(1,2))
    box_square = hv.Path([hv.Bounds(tuple([boxes[box_idx,0],median_projection.shape[0]-boxes[box_idx,3],boxes[box_idx,2],
                            median_projection.shape[0]-boxes[box_idx,1]]))]).options(color='lime')
    return (roi_bounds*box_square).opts(width=600,height=600)

def plot_boxes_traces(data_path,Score,x,y):
    '''Function used as input for the dynamic map in function boxes_exploration_interactive():
    returns the mean(calculated inside the box) flouroescence in time of the selected
    cell. Selection is done by the user by clicking on the box'''
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = get_files_names(data_path)
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    with open(boxes_path,'rb') as f:
        boxes = pickle.load(f)
        f.close()
    boxes = boxes[boxes[:,4]>Score].astype('int')

    if None not in [x,y]:
        try:
            box_idx = [i for i,box in enumerate(boxes) if x<box[2] and x>box[0] and y<(median_projection.shape[0]-box[1])
                    and y>(median_projection.shape[0]-box[3])][0]
        except:
            pass
    else:
        box_idx = 0
    box_trace = images[:,boxes[box_idx,1]:boxes[box_idx,3],boxes[box_idx,0]:boxes[box_idx,2]].mean(axis=(1,2))
    return hv.Curve((np.linspace(0,len(box_trace)-1,len(box_trace)),box_trace),'Frame','Mean box Fluorescence')
            #+hv.Curve((np.linspace(0,len(box_trace)-1,len(box_trace)),caiman_df_f),'Frame','DF/F CaImAn')
            #)#.cols(1)#.opts(width=600)

def plot_seeded_traces(data_path,cnm,Score,x,y,centers):
    '''Function used as input for the dynamic map in function boxes_exploration_interactive():
    returns the DF/F trace calculated with caiman of the corresponding factor of the selected box.
    Selection is done by the user by clicking on the box'''
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = get_files_names(data_path)
    with open(boxes_path,'rb') as f:
        boxes = pickle.load(f)
        f.close()
    boxes = boxes[boxes[:,4]>Score].astype('int')
    if None not in [x,y]:
        try:
            box_idx = [i for i,box in enumerate(boxes) if x<box[2] and x>box[0] and y<(median_projection.shape[0]-box[1])
                    and y>(median_projection.shape[0]-box[3])][0]
        except:
            pass
    else:
        box_idx = 0
    idx_factor = [i for i,center in enumerate(centers) if center[0]>boxes[box_idx,1] and
     center[0]<boxes[box_idx,3] and center[1]>boxes[box_idx,0] and center[1]<boxes[box_idx,2]]
    # in case there is more than one center inside the box choose the one closer to the center of the box
    if len(idx_factor)>1:
        idx_factor = [idx_factor[np.argmin([np.linalg.norm([(boxes[box_idx,3]-boxes[box_idx,1])/2,
                                (boxes[box_idx,2]-boxes[box_idx,0])/2]-c) for c in centers[idx_factor]])]]

    #caiman_df_f = cnm.estimates.F_dff[idx_factor[0]]
    caiman_df_f = cnm.estimates.C[idx_factor[0]]
    denoised_trace = np.max(caiman_df_f)*(cnm.estimates.C[idx_factor[0]]-(np.min(cnm.estimates.C[idx_factor[0]])
                    +np.mean(caiman_df_f)))/(np.max(cnm.estimates.C[idx_factor[0]])-np.min(cnm.estimates.C[idx_factor[0]]))
    #return hv.Curve((np.linspace(0,len(caiman_df_f)-1,len(caiman_df_f)),caiman_df_f),'Frame','DF/F CaImAn')*hv.Curve(denoised_trace)
    return hv.Curve((np.linspace(0,len(caiman_df_f)-1,len(caiman_df_f)),cnm.estimates.C[idx_factor[0]]),'Frame','Denoised')

def boxes_exploration(data_path):
    '''Returns an interactive plot with the agonia boxes with a confidence value above
    a Score selected by the user with a slider. The user can see the mean-trace of a
    specific box and scroll through the boxes with a slider'''
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = get_files_names(data_path)
    with open(boxes_path,'rb') as f:
        boxes = pickle.load(f)
        f.close()

    kdims=[hv.Dimension('Score', values=np.arange(0.05,1,0.05)),
        hv.Dimension('box_idx',values=np.arange(0,len(boxes),1))]
    Experiment = Stream.define('Experiment', data_path=data_path)
    dmap = hv.DynamicMap(plot_AGonia_boxes, kdims=kdims,streams=[Experiment()])
    return dmap

def boxes_exploration_interactive(data_path):
    '''Returns an interactive plot with the agonia boxes with a confidence value above
    a Score selected by the user with a slider. The user can select a box by clicking on it
    and mean box Fluorescence and the Caiman DF/F of such box will be ploted.'''
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = get_files_names(data_path)
    cnm = cnmf.load_CNMF(results_caiman_path)
    img = hv.Image(median_projection,bounds=(0,0,median_projection.shape[1],median_projection.shape[0]
                                                          )).options(cmap='gray')
    with open(boxes_path,'rb') as f:
        boxes = pickle.load(f)
        f.close()

    centers = np.empty((cnm.estimates.A.shape[1],2))
    for i,factor in enumerate(cnm.estimates.A.T):
        centers[i] = center_of_mass(factor.toarray().reshape(cnm.estimates.dims,order='F'))
    #scatter =  hv.Scatter((centers[:,1], median_projection.shape[0] - centers[:,0]))
    kdims=[hv.Dimension('Score', values=np.arange(0.05,1,0.05))]
    tap = streams.SingleTap(transient=True,source=img)
    Experiment = Stream.define('Experiment', data_path=data_path)
    Centers = Stream.define('Centers', centers=centers)
    CaImAn_detection = Stream.define('CaImAn_detection', cnm=cnm)
    dmap = hv.DynamicMap(plot_AGonia_boxes_interactive, kdims=kdims,streams=[Experiment(),tap])
    dmap1 = hv.DynamicMap(plot_boxes_traces, kdims=kdims,streams=[Experiment(),tap])
    dmap2 = hv.DynamicMap(plot_seeded_traces,kdims=kdims,streams=[CaImAn_detection(), Experiment(),tap,Centers()])
    return ((img*dmap).opts(width=500,height=500)+dmap1.opts(width=500,height=250)+dmap2.opts(width=500,height=250)).opts(opts.Curve(framewise=True)).cols(1)

def read_dlc_results(dlc_results_path):
    result = pd.read_hdf(dlc_results_path)
    scorer = result.columns.get_level_values(0)[0]
    return result[scorer]

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


def substract_neuropil(data_path,agonia_th,neuropil_pctl,signal_pctl):
    '''Calculate the neuropil trace as the average fluorescence of every pixel that is not in a box.
    subtract it to every box trace, calculated as the activity of the pixels above the signal_pctl.

    Parameters
    ----------
    data_path : string
        path to folder containing the data
    agonia_th : float
        threshold for detection confidence for each box
    neuropil_pctl : int
        percentile of pixels intensity bellow which pixel is considered to calculate neuropil
    signal_pctl : int
        percentile of pixels intensity above which pixel is considered to calculate trace
    Returns
    -------
    denoised_traces : NxT ndarray
        temporal traces after neuropil subtraction for the N boxes
    neuropil_trace : array
        normalized temporal trace of background noise, length T
    neuropil_power : float
        mean of neuropil trace before normalization
    '''

    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = get_files_names(data_path)
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load Caiman results
    cnm = cnmf.load_CNMF(results_caiman_path)
    # calculate the centers of the CaImAn factors
    centers = np.empty((cnm.estimates.A.shape[1],2))
    for i,factor in enumerate(cnm.estimates.A.T):
        centers[i] = center_of_mass(factor.toarray().reshape(cnm.estimates.dims,order='F'))

    with open(boxes_path,'rb') as f:
        boxes = pickle.load(f)
        f.close()
    # keep only cells above confidence threshold
    boxes = boxes[boxes[:,4]>agonia_th].astype('int')
    #delete boxes that do not have a caiman cell inside
    k = 0
    for cell,box in enumerate(boxes):
        idx_factor = [i for i,center in enumerate(centers) if center[0]>box[1] and
         center[0]<box[3] and center[1]>box[0] and center[1]<box[2]]
        if not idx_factor:
            boxes = np.delete(boxes,cell-k,axis=0)
            k += 1

    boxes_traces = np.empty((boxes.shape[0],images.shape[0]))
    mask = np.zeros(images.shape[1:],dtype=bool)

    for cell,box in enumerate(boxes):
        #boxes_traces[cell] = images[:,box[1]:box[3],box[0]:box[2]].mean(axis=(1,2))
        med = np.median(images[:,box[1]:box[3],box[0]:box[2]],axis=0)
        box_trace = images[:,box[1]:box[3],box[0]:box[2]]
        boxes_traces[cell] = box_trace[:,med>np.percentile(med,signal_pctl)].mean(axis=1)
        mask[box[1].astype('int'):box[3].astype('int'),box[0].astype('int'):box[2].astype('int')]=1

    mask = (1-mask).astype('bool')
    not_cell = images[:,mask]

    #not_cell_med = np.median(not_cell,axis=0)
    #neuropil_trace = not_cell[:,not_cell_med<np.percentile(not_cell_med,neuropil_pctl)].mean(axis=1)
    #denoised_traces = boxes_traces-neuropil_trace

    neuropil_trace = not_cell.mean(axis=1)
    neuropil_power = neuropil_trace.mean()
    neuropil_trace = neuropil_trace/neuropil_power
    denoised_traces =np.array([boxes_traces[i]-neuropil_trace*boxes_traces[i].mean() for i in range(boxes_traces.shape[0])])

    return denoised_traces, neuropil_trace, neuropil_power


def event_overlap(caiman_traces,denoised_agonia,method='dombeck',n_std=2,start_nSigma_bsl=7,stop_nSigma_bsl=5):
    '''binzarize traces (events=1 else=0), then check if events in traces_0 are
    similar to events in traces_1
    Parameters
    ----------
    method : str, 'hard_th' or 'dombeck'
        if hard_th events are everything above mean + n_std*std. If dombeck, uses
        function given by Seba to select events as in paper from dombeck
    n_std : float
        parameter for method hard_th
    start_nSigma_bsl, stop_nSigma_bsl : float
        parameters for method dombeck
    Returns
    -------
    events_overlap : percentage of time in which events of both traces are
        overlaped out of the total event-time
    '''
    if traces_0.shape != traces_1.shape:
        print('Error: traces must be same length, and same number of factors')
    else:
        events_overlap = np.zeros(caiman_traces.shape[0])
        for i,trace in enumerate(caiman_traces):
            if method==hard_th:
                trace_1 = denoised_agonia[i].copy()
                trace_0 = trace.copy()

                STD_0 = np.std(trace_0)
                M_0 = trace_0.mean()
                TH_0 = M_0+n_std*STD_0
                trace_0[trace_0<TH_0]=0
                trace_0[trace_0>=TH_0]=1

                STD_1 = np.std(trace_1)
                M_1 = trace_1.mean()
                TH_1 = M_1+n_std*STD_1
                trace_1[trace_1<TH_1]=0
                trace_1[trace_1>=TH_1]=1
            elif method=='dombeck':
                (trace_0, _, _, _, _, _) = sut.eventFinder(
                trace = trace, start_nSigma_bsl = start_nSigma_bsl, stop_nSigma_bsl = stop_nSigma_bsl,
                FPS = 3, minimumDuration = .3, debugPlot = False)
                trace_0[trace_0!=0]=1

                (trace_1, _, _, _, _, _) = sut.eventFinder(
                trace = denoised_agonia[i]-np.percentile(denoised_agonia[i],10), start_nSigma_bsl = 4, stop_nSigma_bsl = 3,
                FPS = 3, minimumDuration = .3, debugPlot = False)
                trace_1[trace_1!=0]=1

            sum_events = trace_0+trace_1
            sum_events[sum_events==2]=1

            events_overlap[i] = sum(trace_0*trace_1)/sum(sum_events)

        return events_overlap


def event_periods_correlation(caiman_trace,ago_denoised,method='dombeck',n_std=2,start_nSigma_bsl=7,stop_nSigma_bsl=5):
    '''select only event periods and correlate traces denoised with different methods
    Parameters
    ----------
    method : str, 'hard_th' or 'dombeck'
        if hard_th events are everything above mean + n_std*std. If dombeck, uses
        function given by Seba to select events as in paper from dombeck
    n_std : float
        parameter for method hard_th
    start_nSigma_bsl, stop_nSigma_bsl : float
        parameters for method dombeck
    Returns
    -------
    events_corr : correlation of signal periods detected as events
    '''
    if caiman_trace.shape != ago_denoised.shape:
        print('Error: traces must be same length, and same number of factors')
    else:
        events_corr = np.zeros(caiman_trace.shape[0])
        for i,trace in enumerate(caiman_trace):
            if method == hard_th:
                trace_1 = ago_denoised[i].copy()
                trace_1 = (trace_1-min(trace_1))/max(trace_1-min(trace_1))
                trace_0 = trace/max(trace)

                STD_0 = np.std(trace_0)
                M_0 = trace_0.mean()
                TH_0 = M_0+n_std*STD_0
                #trace_1[trace_0<TH_0]=np.nan
                #trace_0[trace_0<TH_0]=np.nan
            elif method == 'dombeck':
                (pos_events_trace_C, _, _, _, _, _) = sut.eventFinder(
                trace = trace, start_nSigma_bsl = start_nSigma_bsl, stop_nSigma_bsl = stop_nSigma_bsl,
                FPS = 3, minimumDuration = .3, debugPlot = False)

            events_corr[i] = np.corrcoef([trace_0[trace_0>TH_0],trace_1[trace_0>TH_0]])[1,0]

    return events_corr

def localvsglobal_neuropil(data_path,agonia_th,only_neuropil_trace=False):
    '''Calculate for each cell the local neuropil trace as the average intensity
    trace of all the surrounding pixels of box that do not belong to another box
    Where surrounding meaning pixel outside the box and inside a box with double
    width and double height and same center.
    Parameters
    ----------
    data_path : string
        path to folder containing the data
    agonia_th : float
        threshold for detection confidence for each box
    Returns
    -------
    local_global_corr : array with local vs global noise correlation for each cell
    '''
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = get_files_names(data_path)
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')

    with open(boxes_path,'rb') as f:
        boxes = pickle.load(f)
        f.close()
    # keep only cells above confidence threshold
    boxes = boxes[boxes[:,4]>agonia_th].astype('int')

    mask = np.zeros(images.shape[1:],dtype=bool)
    for cell,box in enumerate(boxes):
        mask[box[1].astype('int'):box[3].astype('int'),box[0].astype('int'):box[2].astype('int')]=1

    mask = (1-mask).astype('bool')
    not_cell = images[:,mask]
    neuropil_trace = not_cell.mean(axis=1)
    local_global_corr = np.zeros(boxes.shape[0])
    if not only_neuropil_trace:
        for cell,box in enumerate(boxes):
            half_width_x = ((box[3]-box[1])/2).astype('int')
            half_width_y = ((box[2]-box[0])/2).astype('int')
            x_0 = [box[1]-half_width_x if (box[1]-half_width_x)>=0 else 0][0]
            y_0 = [box[0]-half_width_y if (box[0]-half_width_y)>=0 else 0][0]
            local_mask = mask[x_0:box[3]+half_width_x,y_0:box[2]+half_width_y]
            big_box = images[:,x_0:box[3]+half_width_x,y_0:box[2]+half_width_y]
            local_noise = big_box[:,local_mask].mean(axis=1)
            local_global_corr[cell] = np.corrcoef([local_noise,neuropil_trace])[1,0]

    return local_global_corr,neuropil_trace

def signal_to_noise(data_path,agonia_th,ground_truth=None,neurofinder=False):
    '''Calculate signal to noise ratio of each box
    Parameters
    ----------
    data_path : string
        path to folder containing the data
    agonia_th : float
        threshold for detection confidence for each box'''
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = get_files_names(data_path)
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load Caiman results
    cnm = cnmf.load_CNMF(results_caiman_path)
    # calculate the centers of the CaImAn factors
    centers = np.empty((cnm.estimates.A.shape[1],2))
    for i,factor in enumerate(cnm.estimates.A.T):
        centers[i] = center_of_mass(factor.toarray().reshape(cnm.estimates.dims,order='F'))

    with open(boxes_path,'rb') as f:
        boxes = pickle.load(f)
        f.close()
        # keep only cells above confidence threshold
    boxes = boxes[boxes[:,4]>agonia_th].astype('int')

    k = 0
    for cell,box in enumerate(boxes):
        idx_factor = [i for i,center in enumerate(centers) if center[0]>box[1] and
        center[0]<box[3] and center[1]>box[0] and center[1]<box[2]]
        if not idx_factor:
            boxes = np.delete(boxes,cell-k,axis=0)
            k += 1

    if ground_truth is not None:
        stats = ag.compute_stats(np.array(ground_truth), boxes, iou_threshold=0.5)
    stnr = []#np.zeros(boxes.shape[0])
    if neurofinder:
        for cell,box in enumerate(boxes):
            if ground_truth is not None:
                if cell in stats['true_positives'][:,1]:
                    box_trace = images[:,box[1]:box[3],box[0]:box[2]]
                    box_trace_arr = np.array(images[:,box[1]:box[3],box[0]:box[2]])
                    box_trace_arr_cut = box_trace_arr.copy()
                    box_trace_arr_cut[box_trace<30] = np.nan
                    mean_box = np.nanmean(box_trace_arr_cut,axis=(1,2))
                    trace_flat = box_trace[box_trace>30]
                    noise_box = np.std(trace_flat[trace_flat<np.percentile(trace_flat,25)])
                    amp_trace = max(mean_box)-min(mean_box)
                    stnr.append(amp_trace/noise_box)
            else:
                box_trace = images[:,box[1]:box[3],box[0]:box[2]]
                box_trace_arr = np.array(images[:,box[1]:box[3],box[0]:box[2]])
                box_trace_arr_cut = box_trace_arr.copy()
                box_trace_arr_cut[box_trace<30] = np.nan
                mean_box = np.nanmean(box_trace_arr_cut,axis=(1,2))
                trace_flat = box_trace[box_trace>30]
                noise_box = np.std(trace_flat[trace_flat<np.percentile(trace_flat,25)])
                amp_trace = max(mean_box)-min(mean_box)
                stnr.append(amp_trace/noise_box)
    else:
        for cell,box in enumerate(boxes):
            if ground_truth is not None:
                if cell in stats['true_positives'][:,1]:
                    box_trace = images[:,box[1]:box[3],box[0]:box[2]]
                    mean_box = box_trace.mean(axis=(1,2))
                    amp_trace = max(mean_box)-min(mean_box)
                    noise_box = np.std(box_trace[box_trace<np.percentile(box_trace[box_trace!=0],25)])
                    stnr.append(amp_trace/noise_box)
            else:
                box_trace = images[:,box[1]:box[3],box[0]:box[2]]
                mean_box = box_trace.mean(axis=(1,2))
                amp_trace = max(mean_box)-min(mean_box)
                noise_box = np.std(box_trace[box_trace<np.percentile(box_trace[box_trace!=0],25)])
                stnr.append(amp_trace/noise_box)


    return np.array(stnr)

def signal_to_noise_pixel(data_path,agonia_th):
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = get_files_names(data_path)
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')

    with open(boxes_path,'rb') as f:
        boxes = pickle.load(f)
        f.close()
    # keep only cells above confidence threshold
    boxes = boxes[boxes[:,4]>agonia_th].astype('int')
    stnr = np.zeros(boxes.shape[0])

    return stnr

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
        self.data_with_bg=np.full((1,1),np.nan,np.double)
        self.mima=(pmin,pmax)
        self.pool=Pool(workers)

    def extract(self, frame, boxes):
        new_frame=np.full((1,self.data.shape[1]), np.nan, np.double)
        self.data=np.concatenate((self.data,new_frame),axis=0)
        self.data_with_bg=np.concatenate((self.data_with_bg,new_frame),axis=0)
        if len(boxes):
            if len(boxes)-self.data.shape[1]+1>0:
                new_cells=np.full((self.data.shape[0],len(boxes)-self.data.shape[1]+1), np.nan, np.double)
                self.data=np.concatenate((self.data, new_cells), axis=1)
                self.data_with_bg=np.concatenate((self.data_with_bg, new_cells), axis=1)
            patches=[]
            bg_frame=frame.astype(np.double, copy=True)
            for b in boxes:
                bg_frame[int(b[1]):int(b[3]+1),int(b[0]):int(b[2]+1)]=np.nan
                patches.append(frame[int(b[1]):int(b[3]+1),int(b[0]):int(b[2]+1)].flatten())
            traces=self.pool.starmap(_p_extract,[(p,self.mima) for p in patches])
            self.data[-1,1:]=traces
            self.data_with_bg[-1,1:]=traces
            self.data[-1,1:]-=np.nanmean(bg_frame)

    def get_traces(self, tail=0):
        if tail>0 and tail>self.data.shape[0]:
            return self.data[int(-tail):,1:]
        else:
            return self.data[1:,1:],self.data_with_bg[1:,1:]


def traces_extraction_AGONIA(data_path,agonia_th,neuropil_contaminated=False):
    '''Extract traces of boxes using agonia Extractor class'''
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = get_files_names(data_path)
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load Caiman results
    cnm = cnmf.load_CNMF(results_caiman_path)
    # calculate the centers of the CaImAn factors
    centers = np.empty((cnm.estimates.A.shape[1],2))
    for i,factor in enumerate(cnm.estimates.A.T):
        centers[i] = center_of_mass(factor.toarray().reshape(cnm.estimates.dims,order='F'))

    with open(boxes_path,'rb') as f:
        boxes = pickle.load(f)
        f.close()
    # keep only cells above confidence threshold
    boxes = boxes[boxes[:,4]>agonia_th].astype('int')
    k = 0
    for cell,box in enumerate(boxes):
        idx_factor = [i for i,center in enumerate(centers) if center[0]>box[1] and
        center[0]<box[3] and center[1]>box[0] and center[1]<box[2]]
        if not idx_factor:
            boxes = np.delete(boxes,cell-k,axis=0)
            k += 1
    ola = Extractor()
    for frame in images:
        ola.extract(frame,boxes)
    traces,traces_with_bg = ola.get_traces()
    ola.pool.terminate()
    if neuropil_contaminated:
        return traces.T,traces_with_bg.T
    else:
        return traces.T

def distcorr(X, Y):
    '''Compute the distance correlation function'''

    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor








































# end of script
