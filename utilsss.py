import os
os.chdir('/home/pedro/keras-retinanet')
from AGOnIA import AGOnIA
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.components_evaluation import compute_event_exceptionality

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

def agonia_detect(data_path,data_name,median_projection):
    det = AGOnIA('L4+CA1_800.h5',True)
    ROIs = det.detect(median_projection,threshold=0.01,multiplier=2)
    pickle.dump( ROIs, open( os.path.join(data_path,os.path.splitext(data_name[0])[0] + '_boxes.pkl'), "wb" ) )

def seeded_Caiman_wAgonia(data_path,opts):
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

    Ain = np.zeros((np.prod(images.shape[1:]),cajas.shape[0]),dtype=bool)
    for i,box in enumerate(cajas):
        frame = np.zeros(images.shape[1:])
        frame[box[0].astype('int'):box[2].astype('int'),box[1].astype('int'):box[3].astype('int')]=1
        Ain[:,i] = frame.flatten('F')#frame.reshape(np.prod(images.shape[1:])).astype(bool)

    cnm_seeded = cnmf.CNMF(n_processes, params=opts, dview=dview, Ain=Ain)
    try:
        cnm_seeded.fit(images)
        cnm_seeded.save(os.path.join(data_path,os.path.splitext(data_name[0])[0] + '_analysis_results.hdf5'))
    except:
        print('El problema de nuevo...')

    cm.stop_server(dview=dview)

def trace_correlation(data_path, select_cells=False):
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = get_files_names(data_path)
    # load Caiman results
    cnm = cnmf.load_CNMF(results_caiman_path)

    with open(boxes_path,'rb') as f:
        boxes = pickle.load(f)
        f.close()
    boxes = boxes[:,:4].astype('int')

    ### compare temporal traces ###
    # calculate mean over the box and do coefcorr with the caiman trace get a value
    # of correlation for each cell

    # calculate mean for each AGOnIA box
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    boxes_traces = np.empty((boxes.shape[0],images.shape[0]))

    for i,box in enumerate(boxes):
        boxes_traces[i] = images[:,box[0]:box[2],box[1]:box[3]].mean(axis=(1,2))
        boxes_traces[i] = images[:,box[0]:box[2],box[1]:box[3]].mean(axis=(1,2))

    cell_corr = np.empty(len(boxes_traces))
    for cell,Cai in enumerate(cnm.estimates.C):
        cell_corr[cell] = np.corrcoef([Cai,boxes_traces[cell]])[1,0]

    if select_cells:
        #idx_active = [cell for cell,trace in enumerate(boxes_traces) if
        #              np.mean((trace-np.mean(trace))**2)/np.std(trace)**2>active_th]
        fitness, _, _, _ = compute_event_exceptionality(boxes_traces)
        idx_active = [cell for cell,fit in enumerate(fitness) if fit<-20]

    else:
        idx_active = [cell for cell,_ in enumerate(boxes_traces)]

    corr_toplot = [corr for id,corr in enumerate(cell_corr) if id in idx_active and ~np.isnan(corr)]
    fig,ax = plt.subplots()
    p = ax.boxplot(corr_toplot)
    ax.set_ylim([-1,1])
    plt.text(1.2,0.8,'n_cells = {}'.format(len(corr_toplot)))

    return cell_corr, idx_active, boxes_traces












# end of script
