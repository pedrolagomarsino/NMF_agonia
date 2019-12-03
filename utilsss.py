import os
os.chdir('/home/pedro/keras-retinanet')
from AGOnIA import AGOnIA
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning)
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.components_evaluation import compute_event_exceptionality
from scipy.ndimage.measurements import center_of_mass

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
    det = AGOnIA('L4+CA1_800.h5',True)
    ROIs = det.detect(median_projection,threshold=th,multiplier=multiplier)
    pickle.dump( ROIs, open( os.path.join(data_path,os.path.splitext(data_name[0])[0] + '_boxes.pkl'), "wb" ) )

def seeded_Caiman_wAgonia(data_path,opts,agonia_th):
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = get_files_names(data_path)

    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    #load frames in python format (T x X x Y)

    #%% restart cluster to clean up memory
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

    # Load the boxes in the pickle file into the ROIs arrays.
    with open(boxes_path,'rb') as f:
        cajas = pickle.load(f)
        f.close()
    cajas = cajas[cajas[:,4]>agonia_th]

    half_width  = np.mean(cajas[:,2]-cajas[:,0])/2
    half_height = np.mean(cajas[:,3]-cajas[:,1])/2
    gSig= (half_width.astype(int),half_height.astype(int))
    opts_dict = {'gSig': gSig}
    opts.change_params(opts_dict);

    Ain = np.zeros((np.prod(images.shape[1:]),cajas.shape[0]),dtype=bool)
    for i,box in enumerate(cajas):
        frame = np.zeros(images.shape[1:])
        frame[box[1].astype('int'):box[3].astype('int'),box[0].astype('int'):box[2].astype('int')]=1
        Ain[:,i] = frame.flatten('F')#frame.reshape(np.prod(images.shape[1:])).astype(bool)

    cnm_seeded = cnmf.CNMF(n_processes, params=opts, dview=dview, Ain=Ain)
    try:
        cnm_seeded.fit(images)
        cnm_seeded.save(os.path.join(data_path,os.path.splitext(data_name[0])[0] + '_analysis_results.hdf5'))
    except:
        print('El problema de nuevo...')

    cm.stop_server(dview=dview)

def trace_correlation(data_path, agonia_th, select_cells=False,plot_results=True):
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = get_files_names(data_path)
    # load Caiman results
    cnm = cnmf.load_CNMF(results_caiman_path)

    with open(boxes_path,'rb') as f:
        boxes = pickle.load(f)
        f.close()
    boxes = boxes[boxes[:,4]>agonia_th].astype('int')
    boxes.shape
    cnm.estimates.A.shape
    #delete boxes that do not have a caiman cell inside
    k = 0
    for cell,box in enumerate(boxes):
        if np.sum(cnm.estimates.A[:,cell-k].toarray().reshape(cnm.estimates.dims,order='F'
                    )[box[1]:box[3],box[0]:box[2]])==0:
            boxes = np.delete(boxes,cell-k,axis=0)
            k += 1
            print(cell)
    boxes.shape
    k
    ### compare temporal traces ###
    # calculate mean over the box and do coefcorr with the caiman trace get a value
    # of correlation for each cell

    # calculate mean for each AGOnIA box
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    boxes_traces = np.empty((boxes.shape[0],images.shape[0]))

    for i,box in enumerate(boxes):
        boxes_traces[i] = images[:,box[1]:box[3],box[0]:box[2]].mean(axis=(1,2))

    cell_corr = np.empty(len(boxes_traces))
    for cell,Cai in enumerate(cnm.estimates.C):
        cell_corr[cell] = np.corrcoef([Cai,boxes_traces[cell]])[1,0]

    if select_cells:
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

def trace_correlation1(data_path, agonia_th, select_cells=False,plot_results=True):
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = get_files_names(data_path)
    # load Caiman results
    cnm = cnmf.load_CNMF(results_caiman_path)
    centers = np.empty((cnm.estimates.A.shape[1],2))
    for i,factor in enumerate(cnm.estimates.A.T):
        centers[i] = center_of_mass(factor.toarray().reshape(cnm.estimates.dims,order='F'))

    with open(boxes_path,'rb') as f:
        boxes = pickle.load(f)
        f.close()
    boxes = boxes[boxes[:,4]>agonia_th].astype('int')

    #delete boxes that do not have a caiman cell inside
    k = 0
    for cell,box in enumerate(boxes):
        idx_factor = [i for i,center in enumerate(centers) if center[0]>box[1] and
         center[0]<box[3] and center[1]>box[0] and center[1]<box[2]]
        if not idx_factor:
            boxes = np.delete(boxes,cell-k,axis=0)
            k += 1

    ### compare temporal traces ###
    # calculate mean over the box and do coefcorr with the caiman trace get a value
    # of correlation for each cell

    # calculate mean for each AGOnIA box
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    boxes_traces = np.empty((boxes.shape[0],images.shape[0]))

    cell_corr = np.empty(len(boxes_traces))
    for cell,box in enumerate(boxes):
        boxes_traces[cell] = images[:,box[1]:box[3],box[0]:box[2]].mean(axis=(1,2))
        idx_factor = [i for i,center in enumerate(centers) if center[0]>box[1] and
         center[0]<box[3] and center[1]>box[0] and center[1]<box[2]]
        if len(idx_factor)>1:
            #print('ups! more than once center inside box %2d'%(cell))
            #choose the one closer to the center of the box
            idx_factor = [idx_factor[np.argmin([np.linalg.norm([(box[3]-box[1])/2,
                                    (box[2]-box[0])/2]-c) for c in centers[idx_factor]])]]

        cell_corr[cell] = np.corrcoef([cnm.estimates.C[idx_factor[0]],boxes_traces[cell]])[1,0]

    if select_cells:
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



#PATH = '/media/pedro/DATAPART1/AGOnIA/Tiff_samples'
#FOLDERS = os.listdir(PATH)
#data_path = os.path.join(PATH,FOLDERS[2])



























# end of script
