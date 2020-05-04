import os
import sys
import time
import pickle
import imageio
import warnings
sys.path.insert(1,'/home/pedro/keras-retinanet/AGOnIA_release')
sys.path.insert(1,'/home/pedro/Work/AGOnIA/code')
import AGOnIA2 as ag
import numpy as np
import pandas as pd
import seaborn as sns
import holoviews as hv
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.linear_model import LinearRegression
from scipy.ndimage.measurements import center_of_mass
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    warnings.filterwarnings("ignore",category=FutureWarning)
    from caiman.source_extraction.cnmf import cnmf as cnmf
    from caiman.source_extraction.cnmf import params as params
    import utilsss as ut
    import caiman as cm
hv.extension('matplotlib')

PATH = '/media/pedro/DATAPART1/AGOnIA'
results = pickle.load(open(os.path.join(PATH,'complete_analysis_results.pkl'),'rb'))



####################################
### local vs global correlations ###
####################################
all_lVSg = []
for key in results.keys():
    lvsg = np.empty(0)
    for exp in list(results[key]):
        lvsg = np.append(lvsg,results[key][exp]['localVSglobal_neuropil_corr'])
    all_lVSg.append(lvsg)
fig,ax = plt.subplots(figsize=(8,5))
plt.title('LocalvsGlobal noise Correlations',fontsize=15)
plt.boxplot(all_lVSg)
ax.set_xticklabels(results.keys())
plt.savefig(os.path.join(PATH,'LocalVSglobal_neuropil_corr.svg'),format='svg')
plt.show()
results.keys()
[np.median(val) for val in all_lVSg]  #
[np.std(val) for val in all_lVSg]
np.median([item for sublist in all_lVSg for item in sublist])
np.std([item for sublist in all_lVSg for item in sublist])

######################################################
### AGonia vs Caiman background trace correlations ###
######################################################
background_corr_all = pickle.load(open(os.path.join(PATH,"bg_AGvsCai_correlation.pkl"),"rb"))
VPM_corr = pickle.load(open('/media/pedro/DATAPART1/AGOnIA/VPM_corrected/bg_AGvsCai_correlation.pkl','rb'))
background_corr_all.append(VPM_corr)
# background_corr_all = []
order = ['CA1_Ch1','CA1_Ch2','L4','neurofinder','neurofinder_test','ABO','VPM']
#
# full_data_path = '/media/pedro/DATAPART1/AGOnIA/PNAS_RAW_tifs'
# luca_params = pd.read_csv('/media/pedro/DATAPART1/AGOnIA/params/ABOb V3_log.txt',sep=" ")
# luca_params.set_index('name',inplace=True)
# background_corr = np.empty(len(next(os.walk(full_data_path))[1]))
# for i,video in enumerate(next(os.walk(full_data_path))[1]):
#     print(video)
#     data_path = os.path.join(full_data_path,video)
#     data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)
#     agonia_th = np.float(luca_params.loc[video+'.bmp'].sco)
#     seeded = cnmf.load_CNMF(results_caiman_path)
#     _,neuropil_trace = ut.localvsglobal_neuropil(data_path,agonia_th,only_neuropil_trace=True)
#     background_corr[i] = np.corrcoef(neuropil_trace,seeded.estimates.f[0])[0,1]
# background_corr_all.append(background_corr)
# background_corr_all
# pickle.dump(background_corr_all,open(os.path.join(PATH,"bg_AGvsCai_correlation.pkl"),"wb"))

fig,ax = plt.subplots(figsize=(10,5))
plt.boxplot(background_corr_all)
plt.ylabel('Background component correlation AGOnIA vs CaImAn')
ax.set_xticklabels(order)
plt.savefig(os.path.join(PATH,'bg_AGvsCai_correlation.svg'),format='svg')
plt.show()

[np.median(val) for val in background_corr_all]

[np.std(val) for val in background_corr_all]


np.mean([item for sublist in background_corr_all for item in sublist])
np.std([item for sublist in background_corr_all for item in sublist])

####################################################
### Cross correlation between cells and neuropil ###
####################################################
all_cross_corr = []
all_cross_corr_with_bg = []
for key in list(results.keys()):
    cross_corr = np.empty(0)
    cross_corr_with_bg = np.empty(0)
    for exp in list(results[key]):
        indexes = np.triu_indices(results[key][exp]['correlation_matrix_agonia_traces'].shape[0],k=1)
        cross_corr = np.append(cross_corr,results[key][exp]['correlation_matrix_agonia_traces'][indexes])
        cross_corr_with_bg = np.append(cross_corr_with_bg,results[key][exp]['correlation_matrix_agonia_traces_with_bg'][indexes])
    all_cross_corr.append(cross_corr)
    all_cross_corr_with_bg.append(cross_corr_with_bg)
all_toplot=[]
for i,corr in enumerate(all_cross_corr):
    all_toplot.append(all_cross_corr_with_bg[i][~np.isnan(all_cross_corr_with_bg[i])])
    all_toplot.append(corr)

fig,ax = plt.subplots(figsize=(16,5))
plt.boxplot(all_toplot)
plt.ylabel('Cells cross correlation w/without bg subtraction')
ax.set_xticklabels(['CA1_Ch1_wbg','CA1_Ch1','CA1_Ch2_wbg','CA1_Ch2','L4_wbg','L4','NF_wbg','NF','NF_test_wbg','NF_test','ABO_wbg','ABO','VPM_wbg','VPM'])
plt.savefig(os.path.join(PATH,'bg_subtraction_correlations.svg'),format='svg')
plt.show()
for p in range(6):
    print(stats.ttest_ind(all_toplot[2*p],all_toplot[2*p+1],equal_var=False,nan_policy='omit'))
for p in range(6):
    print(stats.ttest_rel(all_toplot[2*p],all_toplot[2*p+1],nan_policy='omit'))
correlation_drop=[]
for p in range(6):
    correlation_drop.append(np.median(all_toplot[p*2])-np.median(all_toplot[p*2+1]))
np.mean(correlation_drop)

[np.mean(val) for val in all_cross_corr]
[np.nanmean(val) for val in all_cross_corr_with_bg]

#################################
### covariance matrix example ###
#################################
key = 'ABO'
exp = '501729039'
fig,axs = plt.subplots(1,2,figsize=(16,8))
fig.suptitle('Covariance Matrix',size=20)
axs[0].imshow(results[key][exp]['correlation_matrix_agonia_traces_with_bg'])
axs[0].set_title('Before background subtraction',size=15)
axs[1].imshow(results[key][exp]['correlation_matrix_agonia_traces'])
axs[1].set_title('After background subtraction',size=15)
plt.savefig(os.path.join(PATH,'Correlation_matrix_example.svg'),format='svg')
plt.show()


#














































#
