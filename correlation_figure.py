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


PATH = '/media/pedro/DATAPART1/AGOnIA'
data_paths = ['CA1/Ch1','CA1/Ch2','L4','neurofinder','neurofinder_test','PNAS_RAW_tifs']
llaves = ['CA1_Ch1','CA1_Ch2','L4','neurofinder','neurofinder_test','ABO']
if os.path.exists(os.path.join(PATH,'complete_figure_dict.pkl')):
    results = pickle.load(open(os.path.join(PATH,'complete_analysis_results.pkl'),'rb'))
else:
    results = {}
    for i,path in enumerate(data_paths):
        data_path = os.path.join(PATH,path)
        dict = pickle.load(open(os.path.join(data_path,llaves[i]+'analysis_results.pkl'),'rb'))
        results.update(dict)
    pickle.dump(results,open(os.path.join(PATH,"complete_analysis_results.pkl"),"wb"))

######################
### Example traces ###
######################
PATHS = ['/media/pedro/DATAPART1/AGOnIA/CA1/Ch1/TSeries-03072019-1203-1210_Ch1__movie_corrected_aligned',
         '/media/pedro/DATAPART1/AGOnIA/L4/1118',
         '/media/pedro/DATAPART1/AGOnIA/neurofinder/neurofinder.01.00',
         '/media/pedro/DATAPART1/AGOnIA/PNAS_RAW_tifs/503109347']
cells = [3,20,57,118]
agonia_th = [.25,0.2,.15,0.05]

caiman_trace = []
agonia_trace = []

for i,data_path in enumerate(PATHS):
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)
    seeded = cnmf.load_CNMF(results_caiman_path)
    traces = ut.traces_extraction_AGONIA(data_path,agonia_th[i])
    caiman_trace.append(seeded.estimates.C[cells[i]])
    agonia_trace.append(traces[cells[i]])
traces_neurofinder = traces.copy()
traces_ABO = traces.copy()
agonia_trace.append(traces_neurofinder[cells[2]])
agonia_trace.append(traces_ABO[cells[3]])

fig,axs = plt.subplots(4,figsize=(16,8))
for i in range(4):
    axs[i].plot(caiman_trace[i])
# add a big axes, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid(False)
plt.xlabel('Frames',size=20)
plt.ylabel('a.u',size=20)
plt.title('Seeded-CNMF traces',size=20)
plt.savefig('/media/pedro/DATAPART1/AGOnIA/example_traces_caiman.svg',format='svg')
fig,axs = plt.subplots(4,figsize=(16,8))
for i in range(4):
    axs[i].plot(agonia_trace[i])
# add a big axes, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid(False)
plt.xlabel('Frames',size=20)
plt.ylabel('a.u',size=20)
plt.title('AGOnIA traces',size=20)
plt.savefig('/media/pedro/DATAPART1/AGOnIA/example_traces_agonia.svg',format='svg')
##########################
### correlation figure ###
##########################

corravsc_all = []
for key in results.keys():
    corr = np.empty(0)
    for exp in list(results[key]):
        corr = np.append(corr,results[key][exp]['corrs_comp_denoised'])
    corravsc_all.append(corr)

fig,ax = plt.subplots(figsize=(8,5))
plt.title('AgoniaVsCaiman Trace Correlations',fontsize=15)
plt.boxplot(corravsc_all)
ax.set_xticklabels(results.keys())
plt.savefig(os.path.join(PATH,'Correlation_agonia_vs_Caiman.svg'),format='svg')
plt.show()

###################################################
### correlation vs signal to noise ratio figure ###
###################################################
stnr_all = []
for key in results.keys():
    stnr = np.empty(0)
    for exp in list(results[key]):
        stnr = np.append(stnr,results[key][exp]['sig_to_noise'])
    stnr_all.append(stnr)
plt.figure(figsize=(8,5))
legend = []
for i,key in enumerate(results.keys()):
    plt.plot(np.log(stnr_all[-i-1]),corravsc_all[-i-1],'.',alpha=.5)
    legend.append(list(results.keys())[-i-1])
plt.legend(legend)#,loc ='lower right')
plt.xlabel('log(Signal to noise ratio)')
plt.ylabel('CaimanVSAgonia trace correlation')
plt.savefig(os.path.join(PATH,'STNR_VS_Correlation.svg'),format='svg')
plt.show()

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
