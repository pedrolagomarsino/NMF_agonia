# Supplementary figure for AGOnIA paper, comparison with CaImAn
import os
import sys
import time
import pickle
import warnings
import scipy.stats
import numpy as np
import pandas as pd
import caiman as cm
import seaborn as sns
import holoviews as hv
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from matplotlib.patches import Rectangle
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import pdist, squareform
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
def distcorr(X, Y):
    """ Compute the distance correlation function

    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
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

#datasets paths
PATH = '/media/pedro/DATAPART1/AGOnIA/datasets_figure'
datasets = {'L4':{'agonia_th' : .2},
            'CA1':{'agonia_th' : .35},
            'ABO':{'agonia_th' : .4},
            'Sofroniew':{'agonia_th' : .2},
            'Svoboda':{'agonia_th' : .2},
            'VPM':{'agonia_th' : .35}
            }
# if the analysis has already been done there should be a pickle with the analysis in a dictionary
datasets = pickle.load(open(os.path.join(PATH,'figure_dict.pkl'),'rb'))

##########################################
########### correlation figure ###########
##########################################

for key in datasets.keys():
    data_path = os.path.join(PATH,key)
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(data_path)
    seeded = cnmf.load_CNMF(results_caiman_path)
    datasets[key]['corrs_comp_denoised'], _, _ = ut.trace_correlation(data_path,
    agonia_th=datasets[key]['agonia_th'],select_cells=False,plot_results=False,denoise=True)
    #datasets[key]['corrs_comp_denoised']=np.array([3,1,2.2])

fig,ax = plt.subplots(figsize=(5,5))
plt.title('MedianVsCaiman Correlations',fontsize=15)
plt.boxplot([datasets['L4']['corrs_comp_denoised'],datasets['CA1']['corrs_comp_denoised'],datasets['ABO']['corrs_comp_denoised'],
             datasets['Sofroniew']['corrs_comp_denoised'],datasets['Svoboda']['corrs_comp_denoised'],datasets['VPM']['corrs_comp_denoised']])
ax.set_xticklabels(datasets.keys())
plt.ylim([0,1])
#plt.savefig(os.path.join(data_path,'correlation_median_or_denoised_weightedvsCaiman'),format='pdf')
plt.savefig(os.path.join(PATH,'correlation_agonia_vs_Caiman.pdf'),format='pdf')
plt.savefig(os.path.join(PATH,'correlation_agonia_vs_Caiman.png'),format='png')
plt.show()

##########################################
## Correlation vs Signal-to-noise ratio ##
##########################################

for key in datasets.keys():
    data_path = os.path.join(PATH,key)
    datasets[key]['sig_to_noise'] = ut.signal_to_noise(data_path,datasets[key]['agonia_th'])

stnr = np.empty(0)
corr = np.empty(0)
for key in datasets.keys():
    if key!='Sofroniew':
        stnr = np.append(stnr,datasets[key]['sig_to_noise'])
        corr = np.append(corr,datasets[key]['corrs_comp_denoised'])

distcorr(stnr,corr)
#np.corrcoef([stnr,corr])[1,0]
#scipy.stats.pearsonr(stnr,corr)
#distcorr(np.log(stnr),corr)
#np.corrcoef([np.log(stnr),corr])[1,0]
#scipy.stats.pearsonr(np.log(stnr),corr)

plt.figure()
legend = []
for key in datasets.keys():
    if key!='Sofroniew':
        plt.plot(np.log(datasets[key]['sig_to_noise']),datasets[key]['corrs_comp_denoised'],'.')
        legend.append(key)
plt.legend(legend)
plt.ylim([0,1.02])
plt.xlabel('log(Signal to noise ratio)')
plt.ylabel('CaimanVSAgonia trace correlation')
plt.savefig(os.path.join(PATH,'STNR_VS_Correlation.pdf'),format='pdf')
plt.savefig(os.path.join(PATH,'STNR_VS_Correlation.png'),format='png')
plt.show()

stnr = datasets['L4']['sig_to_noise']
corr = datasets['L4']['corrs_comp_denoised']
for key in datasets.keys():
    if key!='L4' and key!='Sofroniew':




##########################################
### local vs global noise correlations ###
##########################################

for key in datasets.keys():
    data_path = os.path.join(PATH,key)
    datasets[key]['localVSglobal_neuropil_corr'] = ut.localvsglobal_neuropil(data_path,datasets[key]['agonia_th'])

fig,ax = plt.subplots(figsize=(5,5))
plt.title('LocalvsGlobal noise Correlations',fontsize=15)
plt.boxplot([datasets['L4']['localVSglobal_neuropil_corr'],datasets['CA1']['localVSglobal_neuropil_corr'],
             datasets['ABO']['localVSglobal_neuropil_corr'],datasets['Svoboda']['localVSglobal_neuropil_corr'],
             datasets['VPM']['localVSglobal_neuropil_corr']])
ax.set_xticklabels(legend)
plt.ylim([0,1])
plt.savefig(os.path.join(PATH,'localVSglobal_neuropil_corr.pdf'),format='pdf')
plt.savefig(os.path.join(PATH,'localVSglobal_neuropil_corr.png'),format='png')
plt.show()




pickle.dump(datasets,open(os.path.join(PATH,"figure_dict.pkl"),"wb"))




































#
