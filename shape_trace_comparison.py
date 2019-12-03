import os
import pickle
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    warnings.filterwarnings("ignore",category=FutureWarning)
    import utilsss as ut
import caiman as cm
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import holoviews as hv
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
import scipy.stats
import pandas as pd
from scipy.ndimage.measurements import center_of_mass
hv.extension('matplotlib')


#name of files ordered by agonia params needed
Mesoscopio_names = ['02003', '03001', '06003']
L4_large_names = ['2432', '2433','2414']
L4_names = ['449','447', '448', '486', '488', '857',
             '860', '866', '955', '957', '958', '959',
             '1043', '1044', '1045', '1690-1', '1696',
               '2427','2457', '2458', '2459']
#data_path = '/home/pedro/Work/Hippocampus'
## Data paths
#data_path = '/home/pedro/Work/AGOnIA/Boxes-data/Seeded-Caiman/prueba'
PATH = '/media/pedro/DATAPART1/AGOnIA/Tiff_samples'
FOLDERS = os.listdir(PATH)

for folder in FOLDERS:

    print('Analyzing folder {}'.format(folder))
    if folder in Mesoscopio_names:
        multiplier = 3
        sampling_rate = 30
        agonia_th = 0.3
    elif folder in L4_large_names:
        multiplier = 2
        sampling_rate = 1.5
        agonia_th = 0.2
    elif folder in L4_names:
        multiplier = 1
        sampling_rate = 1.5
        agonia_th = 0.2
    else:
        print('What kind of recording is this? Dont know what multiplier to use')
        break

    data_path = os.path.join(PATH,folder)
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)

    if not results_caiman_path:

        if not boxes_path:
            ## detect with AGOnIA
            ut.agonia_detect(data_path,data_name,median_projection,multiplier=multiplier)

        ## CAIMAN parameters
        # dataset dependent parameters
        #fr = 1/0.758758526502838
        fr = sampling_rate          # imaging rate in frames per second (data obtained from file ....)
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
        gSig = (12,12)#(8, 8)               # expected half size of neurons in pixels
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

        #do the caiman analysis and save results
        try:
            ut.seeded_Caiman_wAgonia(data_path,opts,agonia_th)
        except:
            print('unknown error')

########## correlation between Caiman factors and mean of Agonia boxes##########
M = np.zeros(len(FOLDERS))
MA = np.zeros(len(FOLDERS))
for e,folder in enumerate(FOLDERS):
    print('Analyzing folder {}'.format(folder))
    if folder in Mesoscopio_names:
        agonia_th = 0.3
    elif folder in L4_large_names:
        agonia_th = 0.2
    elif folder in L4_names:
        agonia_th = 0.2
    else:
        print('What kind of recording is this?')
        break
    data_path = os.path.join(PATH,folder)
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)

    if results_caiman_path:
        #corr, idx,boxes_traces = ut.trace_correlation(data_path, agonia_th=agonia_th,
        #select_cells=False,plot_results=True)
        #plt.savefig(os.path.join(data_path,os.path.splitext(data_name[0])[0]+'_all_cells_corr.png'))
        corr_active, idx_active, boxes_traces_active = ut.trace_correlation(data_path,
        agonia_th=agonia_th,select_cells=True,plot_results=False)
        #plt.savefig(os.path.join(data_path,os.path.splitext(data_name[0])[0]+'_active_cells_corr.png'))
        #plt.hist(corr_active)
        #plt.show()
        if idx_active:
            M[e]   = np.nanmean(corr_active)
            MA[e] = np.nanmean(corr_active[idx_active])

p = plt.plot(np.array([1,2]), np.array([M[M!=0],MA[MA!=0]]),'.-')

plt.hist(M[M!=0],np.linspace(0,1,21))
h = plt.hist(MA[MA!=0],np.linspace(0,1,21),alpha=0.5)
data = pd.DataFrame(np.array([M[M!=0],MA[MA!=0]]).T,columns=['All_cells','Active'])
data
data[['All_cells', 'Active']].plot(kind='box',ylim=[0,1])
# significantly higher
scipy.stats.ttest_rel(data['All_cells'], data['Active'])

####################### See boxes and caiman cell centers#####################
data_path = os.path.join(PATH,FOLDERS[2])
data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)
cnm = cnmf.load_CNMF(results_caiman_path)
with open(boxes_path,'rb') as f:
    boxes = pickle.load(f)
    f.close()
boxes = boxes[boxes[:,4]>.3].astype('int')
boxes.shape
cnm.estimates.A.shape
fig,ax = plt.subplots(figsize=(15,15))
ax.imshow(median_projection,'gray')
for box in boxes:
    rect = Rectangle((box[0],box[1]), box[2]-box[0],
          box[3]-box[1],color='r',fill=False)
    ax.add_patch(rect)
for factor in cnm.estimates.A.T:
    dot = center_of_mass(factor.toarray().reshape(cnm.estimates.dims,order='F'))
    plt.plot(dot[1],dot[0],'.',color='yellow')
plt. tight_layout()

################Save median projection with agonia boxes on top#################
for folder in FOLDERS:
    print('Analyzing folder {}'.format(folder))
    if folder in Mesoscopio_names:
        multiplier = 3
        sampling_rate = 30
        agonia_th = 0.3
    elif folder in L4_large_names:
        multiplier = 2
        sampling_rate = 1.5
        agonia_th = 0.2
    elif folder in L4_names:
        multiplier = 1
        sampling_rate = 1.5
        agonia_th = 0.2
    else:
        print('What kind of recording is this? Dont know what multiplier to use')
        break
    data_path = os.path.join(PATH,folder)
    #data_path = '/media/pedro/DATAPART1/AGOnIA/Tiff samles/2433'
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)
    save_path = '/media/pedro/DATAPART1/AGOnIA/Agonia_detection_plots'
    if boxes_path:
        with open(boxes_path,'rb') as f:
            boxes = pickle.load(f)
            f.close()
        boxes = boxes[boxes[:,4]>agonia_th].astype('int')
        #boxes = boxes[:,:4].astype('int')

        fig,ax = plt.subplots(figsize=(15,15))
        ax.imshow(median_projection,'gray')
        for box in boxes:
            rect = Rectangle((box[0],box[1]), box[2]-box[0],
                  box[3]-box[1],color='r',fill=False)
            ax.add_patch(rect)
        plt. tight_layout()
        plt.savefig(os.path.join(save_path,os.path.splitext(data_name[0])[0]+'_agonia_boxes.png'))


################################################################################
data_path = os.path.join(PATH,'2414')
data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)
cnm = cnmf.load_CNMF(results_caiman_path)
corr_active, idx_active, boxes_traces_active = ut.trace_correlation(data_path, agonia_th=0.2,select_cells=False)
idx_active[:5]
corr_active[6]
plt.plot(boxes_traces_active[10])
plt.plot(cnm.estimates.C[10])
cnm.estimates.dims

##########
with open(boxes_path,'rb') as f:
    boxes = pickle.load(f)
    f.close()
cajas = boxes[boxes[:,4]>0.2].astype(int)
pepe = np.array([cajita for cajita in cajas if np.sum(cnm.estimates.A[:,cell].toarray().reshape(cnm.estimates.dims,order='F'
            )[cajita[1]:cajita[3],cajita[0]:cajita[2]])==0])
pepe.shape
cajas.shape
cnm.estimates.C.shape
boxeando = [boxes[j]]
boxes = cajas
k = 0
for cell,box in enumerate(boxes):
    if np.sum(cnm.estimates.A[:,cell-k].toarray().reshape(cnm.estimates.dims,order='F'
                )[box[1]:box[3],box[0]:box[2]])==0:
        boxes = np.delete(boxes,cell-k,axis=0)
        k += 1
    print(cell)
boxes.shape

half_width  = np.mean(cajas[:,2]-cajas[:,0])/2
half_height = np.mean(cajas[:,3]-cajas[:,1])/2
gSig= (half_width.astype(int),half_height.astype(int))
median_projection.shape
Ain = np.zeros((np.prod(median_projection.shape),cajas.shape[0]),dtype=bool)
for i,box in enumerate(cajas):
    frame = np.zeros(median_projection.shape)
    frame[box[1].astype('int'):box[3].astype('int'),box[0].astype('int'):box[2].astype('int')]=1
    Ain[:,i] = frame.flatten('F')#frame.reshape(np.prod(images.shape[1:])).astype(bool)
Ain.shape

#########333
cnm.estimates.C.shape
cajas.shape
cnm.estimates.plot_contours_nb(median_projection)
caiman_cells = [np.reshape(cnm.estimates.A[:,i].toarray(), cnm.estimates.dims, order='F') for i in range(cnm.estimates.A.shape[1])]
fig,ax = plt.subplots(figsize=(10,10))
ax.imshow(np.sum(caiman_cells[21:22],axis=0))
for box in boxes[21:22]:
    rect = Rectangle((box[0],box[1]), box[2]-box[0],
          box[3]-box[1],color='r',fill=False)
    ax.add_patch(rect)
ax.set_title('AGONIA CaImAn detection',fontsize=15)
cell = 21
boxes = boxes.astype('int')
np.sum(cnm.estimates.A[:,cell].toarray().reshape(cnm.estimates.dims,order='F')[boxes[cell,1]:boxes[cell,3],boxes[cell,0]:boxes[cell,2]])
pepe = np.delete(boxes,21,axis=0)
boxes.shape
pepe.shape
fig,ax = plt.subplots(figsize=(10,10))
ax.imshow(median_projection,'gray')
rect = Rectangle((boxes[cell,0],boxes[cell,1]), boxes[cell,2]-boxes[cell,0],
      boxes[cell,3]-boxes[cell,1],color='r',fill=False)
ax.add_patch(rect)
plt.imshow(cnm.estimates.A[:,cell].toarray().reshape(cnm.estimates.dims,order='F'))#[boxes[cell,1]:
                                                boxes[cell,3],boxes[cell,0]:boxes[cell,2]])
cnm.estimates.plot_contours_nb(img=median_projection)
plt.imshow(cnm.estimates.A[:,cell].toarray().reshape(cnm.estimates.dims,order='C'))
cnm.estimates.A[:,0].reshape([324,328])
len(idx_active)
idx_active
plt.plot(boxes_traces[12])

cajas.shape


fig,ax = plt.subplots(figsize=(15,15))
ax.imshow(median_projection,'gray')
for box in cajas:
    rect = Rectangle((box[0],box[1]), box[2]-box[0],
    box[3]-box[1],color='r',fill=False)
    ax.add_patch(rect)


boxes_traces_active.shape

plt.plot(boxes_traces_active[6])
cnm.estimates.C.shape

f, ax = plt.subplots()
ax.plot(cnm.estimates.C[10])
