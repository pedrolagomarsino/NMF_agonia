import os
import sys
import time
import pickle
import imageio
import warnings
sys.path.insert(1,'/home/pedro/keras-retinanet/AGOnIA_release')
import AGOnIA2 as ag
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

PATH = '/media/pedro/DATAPART1/AGOnIA/neurofinder_test'
luca_params = pd.read_csv('/media/pedro/DATAPART1/AGOnIA/params/NEU ALL.txt',sep=" ")
luca_params.set_index('name',inplace=True)
#PATH = '/media/pedro/DATAPART1/AGOnIA/datasets_figure/neurofinder'
for path in next(os.walk(PATH))[1]:
    print('starting analysis of: '+path)
    data_path = os.path.join(PATH,path)
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)
    multiplier = np.float(luca_params.loc[path+'.bmp']['mul'])
    agonia_th = np.float(luca_params.loc[path+'.bmp'].sco)

    ## params
    #fr = 2.5                    # imaging rate in frames per second (data obtained from file ....)
    #decay_time = 0.65           # length of a typical transient in seconds (this is the value given by Marco Brondi, which is different from the one in the original notebook)

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
    #'fr': fr,
    #'decay_time': decay_time,
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
        images = ut.memmap_movie(fnames,load=True)
        # ut.caiman_motion_correct(fnames,opts)
        # data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)
        # Yr, dims, T = cm.load_memmap(fname_new)
        # images = np.reshape(Yr.T, [T] + list(dims), order='F')
        median = np.median(images,axis=0)
        plt.imsave(os.path.join(data_path,'MED_'+os.path.splitext(data_name[0])[0]+'.jpg'),median,cmap='gist_gray')

    # run AGOnIA
    if not boxes_path:
        ## detect with AGOnIA
        data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)
        ut.agonia_detect(data_path,data_name,median_projection,multiplier=multiplier)

    if not results_caiman_path:
        ut.seeded_Caiman_wAgonia(data_path,opts,agonia_th=agonia_th)

# Check results 
data_path = '/media/pedro/DATAPART1/AGOnIA/PNAS_RAW_tifs/539670003'
data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)
seeded = cnmf.load_CNMF(results_caiman_path)
seeded.estimates.nb_view_components(img=median_projection,denoised_color='red')


#datasets paths
full_data_path = PATH#os.path.join(PATH,key,'full_dataset')
anotations_all = pd.read_csv('/media/pedro/DATAPART1/AGOnIA/TestSets_ENH/NEU_all.csv')
anotations_all.columns = ['experiment','xmin','ymin','xmax','ymax','id']
only_true_positives = True

# if the analysis has already been done there should be a pickle with the analysis in a dictionary
key = 'neurofinder_test' #'L4'
if only_true_positives:
    if os.path.exists(os.path.join(PATH,key+'analysis_results_only_truepos.pkl')):
        datasets = pickle.load(open(os.path.join(PATH,key+'analysis_results_only_truepos.pkl'),'rb'))
    else:
        datasets = {key:{}}
else:
    if os.path.exists(os.path.join(PATH,key+'analysis_results.pkl')):
        datasets = pickle.load(open(os.path.join(PATH,key+'analysis_results.pkl'),'rb'))
    else:
        datasets = {key:{}}

##########################################
########### correlation figure ###########
##########################################

for video in next(os.walk(full_data_path))[1]:
    print(video)
    data_path = os.path.join(full_data_path,video)
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)
    agonia_th = np.float(luca_params.loc[video+'.bmp'].sco)
    seeded = cnmf.load_CNMF(results_caiman_path)
    if only_true_positives:
        #load ground truth and calculate true positives
        ground_truth = []
        for index, row in anotations_all.loc[anotations_all['experiment']==video+'.bmp'].iterrows():
            ground_truth.append([row['xmin'],row['ymin'],row['xmax'],row['ymax']])
        #load boxes
        with open(boxes_path,'rb') as f:
            boxes = pickle.load(f)
            f.close()
        # keep only cells above confidence threshold
        boxes = boxes[boxes[:,4]>agonia_th].astype('int')
        # load Caiman results
        #cnm = cnmf.load_CNMF(results_caiman_path)
        # calculate the centers of the CaImAn factors
        centers = np.empty((seeded.estimates.A.shape[1],2))
        for i,factor in enumerate(seeded.estimates.A.T):
            centers[i] = center_of_mass(factor.toarray().reshape(seeded.estimates.dims,order='F'))

        k = 0
        for cell,box in enumerate(boxes):
            idx_factor = [i for i,center in enumerate(centers) if center[0]>box[1] and
            center[0]<box[3] and center[1]>box[0] and center[1]<box[2]]
            if not idx_factor:
                boxes = np.delete(boxes,cell-k,axis=0)
                k += 1
        stats = ag.compute_stats(np.array(ground_truth), boxes, iou_threshold=0.5)

    #seeded = cnmf.load_CNMF(results_caiman_path)
    traces = ut.traces_extraction_AGONIA(data_path,agonia_th)

    if only_true_positives:
        corr = []#np.empty(stats['true_positives'].shape[0])
    else:
        corr = np.empty(traces.shape[0])

    datasets[key][video]={}
    for i,trace in enumerate(traces):
        if only_true_positives:
            if i in stats['true_positives'][:,1]:
                corr.append(np.corrcoef([trace,seeded.estimates.C[i]])[1,0])
        else:
            corr[i] = np.corrcoef([trace,seeded.estimates.C[i]])[1,0]
    datasets[key][video]['corrs_comp_denoised'] = np.array(corr)

corravsc_all = np.empty(0)
corr_avsc = []
labels = []
for exp in list(datasets[key]):
    corravsc_all = np.append(corravsc_all,datasets[key][exp]['corrs_comp_denoised'])
    corr_avsc.append(datasets[key][exp]['corrs_comp_denoised'])
    labels.append(exp[-4:])


fig,ax = plt.subplots(figsize=(8,5))
plt.title('AgoniaVsCaiman Trace Correlations',fontsize=15)
plt.boxplot(corr_avsc)
ax.set_xticklabels(labels)
#plt.ylim([-0.25,1.05])
if only_true_positives:
    plt.savefig(os.path.join(full_data_path,key+'_correlation_agonia_vs_Caiman_only_truepos.png'),format='png')
else:
    plt.savefig(os.path.join(full_data_path,key+'_correlation_agonia_vs_Caiman.png'),format='png')
plt.show()

fig,ax = plt.subplots(figsize=(5,5))
plt.title('AgoniaVsCaiman Trace Correlations',fontsize=15)
plt.boxplot(corravsc_all)
ax.set_xticklabels([key])
#plt.ylim([-0.25,1.05])
if only_true_positives:
    plt.savefig(os.path.join(full_data_path,key+'_all_correlation_agonia_vs_Caiman_only_tp.png'),format='png')
else:
    plt.savefig(os.path.join(full_data_path,key+'_all_correlation_agonia_vs_Caiman.png'),format='png')
plt.show()
##########################################
## Correlation vs Signal-to-noise ratio ##
##########################################

for video in next(os.walk(full_data_path))[1]:
    print(video)
    if only_true_positives:
        ground_truth = []
        for index, row in anotations_all.loc[anotations_all['experiment']==video+'.bmp'].iterrows():
            ground_truth.append([row['xmin'],row['ymin'],row['xmax'],row['ymax']])
    else:
        ground_truth=None
    data_path = os.path.join(full_data_path,video)
    agonia_th = np.float(luca_params.loc[video+'.bmp'].sco)
    datasets[key][video]['sig_to_noise'] = ut.signal_to_noise(data_path,agonia_th,ground_truth,neurofinder=True)

stnr = np.empty(0)
corr = np.empty(0)
for exp in list(datasets[key]):
    #if exp!='neurofinder.02.00' and exp!='neurofinder.02.00.test' and exp!='neurofinder.02.01' and exp!='neurofinder.02.01.test':
    stnr = np.append(stnr,datasets[key][exp]['sig_to_noise'])
    corr = np.append(corr,datasets[key][exp]['corrs_comp_denoised'])

plt.figure(figsize=(8,5))
legend = []
for exp in list(datasets[key]):
    #if exp!='neurofinder.02.00' and exp!='neurofinder.02.00.test' and exp!='neurofinder.02.01' and exp!='neurofinder.02.01.test':
    plt.plot(np.log(datasets[key][exp]['sig_to_noise']),datasets[key][exp]['corrs_comp_denoised'],'.')
    legend.append(exp[-10:])
plt.legend(legend,loc ='lower right')
plt.xlabel('log(Signal to noise ratio)')
plt.ylabel('CaimanVSAgonia trace correlation')

#plt.ylim([-0.25,1.05])
if only_true_positives:
    plt.savefig(os.path.join(full_data_path,key+'_STNR_VS_Correlation_only_truepos.png'),format='png')
else:
    plt.savefig(os.path.join(full_data_path,key+'_STNR_VS_Correlation.png'),format='png')
plt.show()
##########################################
### local vs global noise correlations ###
##########################################
#I think its not worth it to make this comparison for only the true positives
for video in next(os.walk(full_data_path))[1]:
    print(video)
    data_path = os.path.join(full_data_path,video)
    agonia_th = np.float(luca_params.loc[video+'.bmp'].sco)
    datasets[key][video]['localVSglobal_neuropil_corr'] = ut.localvsglobal_neuropil(data_path,agonia_th)

lVSg = []
all_lVSg = np.empty(0)
for exp in list(datasets[key]):
    lVSg.append(datasets[key][exp]['localVSglobal_neuropil_corr'])
    all_lVSg = np.append(all_lVSg,datasets[key][exp]['localVSglobal_neuropil_corr'])
fig,ax = plt.subplots(figsize=(16,5))
plt.title('LocalvsGlobal noise Correlations',fontsize=15)
plt.boxplot(lVSg)
ax.set_xticklabels(labels)
plt.savefig(os.path.join(full_data_path,key+'_localVSglobal_neuropil_corr.png'),format='png')
plt.show()

fig,ax = plt.subplots(figsize=(5,5))
plt.title(key+'LocalvsGlobal noise Correlations',fontsize=15)
plt.boxplot(all_lVSg)
#plt.ylim([0,1])
plt.savefig(os.path.join(full_data_path,key+'_all_localVSglobal_neuropil_corr.png'),format='png')
plt.show()

#save results
if only_true_positives:
    pickle.dump(datasets,open(os.path.join(PATH,key+"analysis_results_only_truepos.pkl"),"wb"))
else:
    pickle.dump(datasets,open(os.path.join(PATH,key+"analysis_results.pkl"),"wb"))






#
