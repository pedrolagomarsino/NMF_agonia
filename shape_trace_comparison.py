import os
import pickle
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import utilsss as ut
import caiman as cm
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import holoviews as hv
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
hv.extension('matplotlib')



## Data paths

#data_path = '/home/pedro/Work/AGOnIA/Boxes-data/Seeded-Caiman/prueba'
PATH = '/media/pedro/DATAPART1/AGOnIA/Tiff_samples'
FOLDERS = os.listdir(PATH)
for folder in FOLDERS:
    print('Analyzing folder {}'.format(folder))
    data_path = os.path.join(PATH,folder)
    #data_path = '/media/pedro/DATAPART1/AGOnIA/Tiff samles/2433'
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)

    if not results_caiman_path:
        ## CAIMAN parameters
        # dataset dependent parameters
        fr = 1/0.758758526502838    # imaging rate in frames per second (data obtained from file ....)
        decay_time = 0.65           # length of a typical transient in seconds (this is the value given by Marco Brondi, which is different from the one in the original notebook)

        # motion correction parameters
        strides = (48, 48)          # start a new patch for pw-rigid motion correction every x pixels
        overlaps = (24, 24)         # overlap between patches (size of patch strides+overlaps)
        max_shifts = (6,6)          # maximum allowed rigid shifts (in pixels)
        max_deviation_rigid = 3     # maximum shifts deviation allowed for patch with respect to rigid shifts
        pw_rigid = True             # flag for performing non-rigid motion correction

        # parameters for source extraction and deconvolution
        p = 1                       # order of the autoregressive system
        gnb = 2                     # number of global background components
        merge_thr = 0.85            # merging threshold, max correlation allowed
        rf = None                   # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
        stride_cnmf = 6             # amount of overlap between the patches in pixels
        K = 4                       # number of components per patch
        gSig = (8, 8)               # expected half size of neurons in pixels
        method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
        ssub = 1                    # spatial subsampling during initialization
        tsub = 1                    # temporal subsampling during intialization

        # parameters for component evaluation
        min_SNR = 2.0               # signal to noise ratio for accepting a component
        rval_thr = 0.85             # space correlation threshold for accepting a component
        cnn_thr = 0.99              # threshold for CNN based classifier
        cnn_lowest = 0.1            # neurons with cnn probability lower than this value are rejected

        opts_dict = {'fnames': fnames,
        'fr': fr,
        'decay_time': decay_time,
        'strides': strides,
        'overlaps': overlaps,
        'max_shifts': max_shifts,
        'max_deviation_rigid': max_deviation_rigid,
        'pw_rigid': pw_rigid,
        'p': p,
        'nb': gnb,
        'rf': rf,
        'K': K,
        'stride': stride_cnmf,
        'method_init': method_init,
        'rolling_sum': True,
        'only_init': False,
        'gSig': gSig,
        'ssub': ssub,
        'tsub': tsub,
        'merge_thr': merge_thr,
        'min_SNR': min_SNR,
        'rval_thr': rval_thr,
        'use_cnn': True,
        'min_cnn_thr': cnn_thr,
        'cnn_lowest': cnn_lowest}

        opts = params.CNMFParams(params_dict=opts_dict)

        if not fname_new:
            #do mmaping
            ut.caiman_motion_correct(fnames,opts)

        if not boxes_path:
            ## detect with AGOnIA
            ut.agonia_detect(data_path,data_name,median_projection)

        #do the caiman analysis and save results
        try:
            ut.seeded_Caiman_wAgonia(data_path,opts)
        except:
            print('unknown error')

#data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)

#corr_1, idx_1 = ut.trace_correlation(data_path,select_cells=True)



for folder in FOLDERS:
    print('Analyzing folder {}'.format(folder))
    data_path = os.path.join(PATH,folder)
    #data_path = '/media/pedro/DATAPART1/AGOnIA/Tiff samles/2433'
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)

    if results_caiman_path:
        corr, idx = ut.trace_correlation(data_path,select_cells=False)
        plt.savefig(os.path.join(data_path,os.path.splitext(data_name[0])[0]+'_all_cells_corr.png'))
        corr_active, idx_active = ut.trace_correlation(data_path,select_cells=True)
        plt.savefig(os.path.join(data_path,os.path.splitext(data_name[0])[0]+'_active_cells_corr.png'))

for folder in FOLDERS:
    print('Analyzing folder {}'.format(folder))
    data_path = os.path.join(PATH,folder)
    #data_path = '/media/pedro/DATAPART1/AGOnIA/Tiff samles/2433'
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)
    save_path = '/media/pedro/DATAPART1/AGOnIA/Agonia_detection_plots'
    if boxes_path:
        with open(boxes_path,'rb') as f:
            boxes = pickle.load(f)
            f.close()
        boxes = boxes[:,:4].astype('int')

        fig,ax = plt.subplots(figsize=(15,15))
        ax.imshow(median_projection,'gray')
        for box in boxes:
            rect = Rectangle((box[0],box[1]), box[2]-box[0],
                  box[3]-box[1],color='r',fill=False)
            ax.add_patch(rect)
        plt. tight_layout()
        plt.savefig(os.path.join(save_path,os.path.splitext(data_name[0])[0]+'_agonia_boxes.png'))



data_path = os.path.join(PATH,'2433')
#data_path = '/media/pedro/DATAPART1/AGOnIA/Tiff samles/2433'
data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)
with open(boxes_path,'rb') as f:
    boxes = pickle.load(f)
    f.close()
boxes = boxes[:,:4].astype('int')
corr_active, idx_active, boxes_traces= ut.trace_correlation(data_path,select_cells=True)

fig,ax = plt.subplots(figsize=(15,15))
ax.imshow(median_projection,'gray')
for id,box in enumerate(boxes):
    if id in idx_active:
        rect = Rectangle((box[0],box[1]), box[2]-box[0],
        box[3]-box[1],color='g',fill=False)
        ax.add_patch(rect)
    else:
        rect = Rectangle((box[0],box[1]), box[2]-box[0],
        box[3]-box[1],color='r',fill=False)
        ax.add_patch(rect)









plt.plot(boxes_traces[idx_active[1]])












#end script
