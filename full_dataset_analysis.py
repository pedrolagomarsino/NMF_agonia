import os
import sys
import time
import pickle
import imageio
import warnings
import numpy as np
import pandas as pd
import caiman as cm
import seaborn as sns
import holoviews as hv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.linear_model import LinearRegression
from scipy.ndimage.measurements import center_of_mass
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    warnings.filterwarnings("ignore",category=FutureWarning)
    import utilsss as ut
hv.extension('matplotlib')

sys.path.insert(1, '/home/pedro/Work/Hippocampus/code')
import to_Pedro as sut

#PATH = '/media/pedro/DATAPART1/AGOnIA/datasets_figure/L4/full_dataset'
PATH = '/media/pedro/DATA_BUCKET/PNAS_RAW_tifs'
for path in os.listdir(PATH)[14:]:
    print('starting analysis of: '+path)
    data_path = os.path.join(PATH,path)
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)

    ## params
    fr = 30                    # imaging rate in frames per second (data obtained from file ....)
    decay_time = 0.65           # length of a typical transient in seconds (this is the value given by Marco Brondi, which is different from the one in the original notebook)

    # motion correction parameters
    strides = (48, 48)          # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)         # overlap between patches (size of patch strides+overlaps)
    max_shifts = (6,6)          # maximum allowed rigid shifts (in pixels)
    max_deviation_rigid = 3     # maximum shifts deviation allowed for patch with respect to rigid shifts
    pw_rigid = True             # flag for performing non-rigid motion correction

    # parameters for source extraction and deconvolution
    p = 0                       # order of the autoregressive system
    gnb = 2                     # number of global background components
    merge_thr = 1#0.85            # merging threshold, max correlation allowed
    rf = None#24                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 15            # amount of overlap between the patches in pixels
    K = 4                       # number of components per patch
    gSig = (8,8)                # expected half size of neurons in pixels
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

    # data preparation
    if not fname_new :
        ut.caiman_motion_correct(fnames,opts)
        data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)
        Yr, dims, T = cm.load_memmap(fname_new)
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        median = np.median(images,axis=0)
        plt.imsave(os.path.join(data_path,'MED_'+os.path.splitext(data_name[0])[0]+'.jpg'),median,cmap='gist_gray')

    # run AGOnIA
    if not boxes_path:
        ## detect with AGOnIA
        data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)
        ut.agonia_detect(data_path,data_name,median_projection,multiplier=2.3)


    agonia_th = .35
    ut.seeded_Caiman_wAgonia(data_path,opts,agonia_th=agonia_th)


data_path = '/media/pedro/DATA_BUCKET/PNAS_RAW_tifs/540684467'
data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)
seeded = cnmf.load_CNMF(results_caiman_path)
seeded.estimates.nb_view_components(img=median_projection,denoised_color='red')

#datasets paths
PATH = '/media/pedro/DATA_BUCKET/PNAS_RAW_tifs/'
datasets = {'L4':{'agonia_th' : .2},
            'CA1':{'agonia_th' : .35},
            'ABO':{'agonia_th' : .4},
            'Sofroniew':{'agonia_th' : .2},
            'Svoboda':{'agonia_th' : .2},
            'VPM':{'agonia_th' : .35}
            }
# if the analysis has already been done there should be a pickle with the analysis in a dictionary
datasets = pickle.load(open(os.path.join(PATH,'figure_dict.pkl'),'rb'))
names = []
for key in datasets.keys():
    if key!='Sofroniew':
        names.append(key)

##########################################
########### correlation figure ###########
##########################################

for key in datasets.keys():
    if key!='Sofroniew':
        print(key)
        key ='ABO'
        full_data_path = PATH#os.path.join(PATH,key,'full_dataset')
        for video in os.listdir(full_data_path)[1:]:
            print(video)
            data_path = os.path.join(full_data_path,video)
            data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)
            seeded = cnmf.load_CNMF(results_caiman_path)
            traces = ut.traces_extraction_AGONIA(data_path,datasets[key]['agonia_th'])
            corr = np.empty(traces.shape[0])
            datasets[key][video]={}
            for i,trace in enumerate(traces):
                corr[i] = np.corrcoef([trace,seeded.estimates.C[i]])[1,0]
            datasets[key][video]['corrs_comp_denoised'] = corr

corravsc_all = np.empty(0)
corr_avsc = []
for exp in list(datasets[key])[1:]:
    corravsc_all = np.append(corravsc_all,datasets[key][exp]['corrs_comp_denoised'])
    corr_avsc.append(datasets[key][exp]['corrs_comp_denoised'])
fig,ax = plt.subplots(figsize=(8,5))
plt.title('AgoniaVsCaiman Trace Correlations',fontsize=15)
plt.boxplot(corr_avsc)
#plt.ylim([0,1])
plt.savefig(os.path.join(full_data_path,'ABO_correlation_agonia_vs_Caiman.png'),format='png')
plt.show()

fig,ax = plt.subplots(figsize=(5,5))
plt.title('AgoniaVsCaiman Trace Correlations',fontsize=15)
plt.boxplot(corravsc_all)
ax.set_xticklabels([key])
plt.savefig(os.path.join(full_data_path,'ABO_all_correlation_agonia_vs_Caiman.png'),format='png')
plt.show()

##########################################
## Correlation vs Signal-to-noise ratio ##
##########################################

for video in next(os.walk(full_data_path))[1]:
    print(video)
    data_path = os.path.join(full_data_path,video)
    datasets[key][video]['sig_to_noise'] = ut.signal_to_noise(data_path,datasets[key]['agonia_th'])

stnr = np.empty(0)
corr = np.empty(0)
for exp in list(datasets[key])[1:]:
    stnr = np.append(stnr,datasets[key][exp]['sig_to_noise'])
    corr = np.append(corr,datasets[key][exp]['corrs_comp_denoised'])

plt.plot(np.log(stnr),corr,'.')
plt.figure(figsize=(8,5))
legend = []
for exp in list(datasets[key])[1:]:
    plt.plot(np.log(datasets[key][exp]['sig_to_noise']),datasets[key][exp]['corrs_comp_denoised'],'.')
    legend.append(exp)
plt.legend(legend)
#plt.ylim([0,1.02])
plt.xlabel('log(Signal to noise ratio)')
plt.ylabel('CaimanVSAgonia trace correlation')
plt.savefig(os.path.join(full_data_path,'ABO_STNR_VS_Correlation.png'),format='png')
plt.show()
##########################################
### local vs global noise correlations ###
##########################################

for video in next(os.walk(full_data_path))[1]:
    print(video)
    data_path = os.path.join(full_data_path,video)
    datasets['L4'][video]['localVSglobal_neuropil_corr'] = ut.localvsglobal_neuropil(data_path,datasets['L4']['agonia_th'])
lVSg = []
all_lVSg = np.empty(0)
for exp in list(datasets['L4'])[1:]:
    lVSg.append(datasets['L4'][exp]['localVSglobal_neuropil_corr'])
    all_lVSg = np.append(all_lVSg,datasets['L4'][exp]['localVSglobal_neuropil_corr'])
fig,ax = plt.subplots(figsize=(8,5))
plt.title('LocalvsGlobal noise Correlations',fontsize=15)
plt.boxplot(lVSg)
ax.set_xticklabels(legend)
plt.ylim([0,1])
plt.savefig(os.path.join(full_data_path,'L4_localVSglobal_neuropil_corr.png'),format='png')
plt.show()

fig,ax = plt.subplots(figsize=(5,5))
plt.title('L4 LocalvsGlobal noise Correlations',fontsize=15)
plt.boxplot(all_lVSg)
plt.ylim([0,1])
plt.savefig(os.path.join(full_data_path,'L4_all_localVSglobal_neuropil_corr.png'),format='png')
plt.show()

pickle.dump(datasets,open(os.path.join(PATH,"full_data_figure_dict.pkl"),"wb"))


































#
