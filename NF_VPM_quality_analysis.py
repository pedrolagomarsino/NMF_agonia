# extra analysis for paper:
# The idea is to show that VPM and NF datasets are qualitatively worst than the rest,
# this is not evident from the analysis made so far, SNR is not lower and correlations are not worst.
# we'll show two things:
# _ zeros issue with NF: many recordings from NF have a bunch of zeros everywhere. Do
#   histograms and maybe medians. see if every recordings or which ones.
# _ perhaps the frequency count histograms of the fluorescence values of the average
#   projection image. save the average + - s.d. of min max and min-max for nf and vpm.
#   do the same for ABO, LIV and CA1. let's see if NF and VPM have significantly
#   different (smaller) dynamic range. 10 frames for ABO, 20 for VPN e 200 for NF
import os

import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    warnings.filterwarnings("ignore",category=FutureWarning)
    import utilsss as ut
    import caiman as cm
PATH = '/media/pedro/DATAPART1/AGOnIA'
data_paths = ['CA1/Ch1','CA1/Ch2','L4','neurofinder','neurofinder_test','PNAS_RAW_tifs','VPM_corrected']
l4 = os.path.join(PATH,data_paths[2],'943')
paths_l4 = [os.path.join(PATH,data_paths[2],name) for name in next(os.walk(os.path.join(PATH,data_paths[2])))[1]]
paths_nf = [os.path.join(PATH,data_paths[3],name) for name in next(os.walk(os.path.join(PATH,data_paths[3])))[1]]
paths_nf_test = [os.path.join(PATH,data_paths[4],name) for name in next(os.walk(os.path.join(PATH,data_paths[4])))[1]]
paths_vpm = [os.path.join(PATH,data_paths[-1],name) for name in next(os.walk(os.path.join(PATH,data_paths[-1])))[1]]
### zero pixels ###
for path in paths_nf:
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(path)
    print((median_projection==0).sum()/(512*512))
x=[0.002178192138671875,0.076385498046875,0.5810966491699219,0.6515998840332031,0.11528396606445312,0.11734390258789062]
np.std([0.002178192138671875,0.076385498046875,0.5810966491699219,0.6515998840332031,0.11528396606445312,0.11734390258789062])
np.sqrt(np.sum([(t-np.mean(x))**2 for t in x])/len(x))

for path in paths_nf_test:
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(path)
    print((median_projection==0).sum()/(512*512))
np.std([0.00194549560546875,0.09865570068359375,0.6284103393554688,0.5911407470703125,0.14682388305664062,0.11803054809570312])

for path in paths_l4:
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(path)
    print((median_projection==0).sum()/(512**2))

for path in paths_vpm:
    data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(path)
    print((median_projection==0).sum()/(196**2))
np.std([0.17531757600999584
,0.23542274052478135
,0.22842044981257809
,0.2890722615576843
,0.348006039150354
,0.3009423157017909
,0.23878071636817993
,0.2726728446480633
,0.24357038733860892])

the_worst = paths_nf[3]
data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(the_worst)
Yr, dims, T = cm.load_memmap(fname_new)
images_nf_worst = np.reshape(Yr.T, [T] + list(dims), order='F')
zeros_frames = []
for im in images_nf_worst:
    zeros_frames.append(np.sum((im<3)))
fig,ax = plt.subplots(figsize=(12,6))
ax.plot(np.array(zeros_frames)/(512**2))
ax.set_ylim([0,1])
ax.set_xlabel('frame',fontsize=20)
ax.set_ylabel('fraction of zeros',fontsize=20)
plt.savefig('/media/pedro/DATAPART1/AGOnIA/fraction_of_zeros_nf_recording.svg')

random_vpm = paths_vpm[0]
data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(random_vpm)
Yr, dims, T = cm.load_memmap(fname_new)
images = np.reshape(Yr.T, [T] + list(dims), order='F')
med = np.median(images,axis=0)
plt.imshow(med)
min(med.flatten())
max(med.flatten())

gaus = [random.gauss(.5,.15) for x in range(10000)]
p = [np.percentile(med,x) for x in range(101)]
gaugau = [np.percentile(gaus,x) for x in range(101)]
pepe = [np.percentile(median_projection_vpm,x) for x in range(101)]
fig,ax = plt.subplots()
ax.plot(p)
ax2=ax.twinx()
ax2.plot(gaugau)

fig,ax = plt.subplots()
ax.plot(p)
ax2=ax.twinx()
ax2.plot(gaugau)

zeros_frames_vpm = []
for im in images:
    zeros_frames_vpm.append(np.sum((im<3)))
fig,ax = plt.subplots(figsize=(12,6))
ax.plot(np.array(zeros_frames_vpm)/(512**2))
ax.set_ylim([0,1])
ax.set_xlabel('frame',fontsize=20)
ax.set_ylabel('fraction of zeros',fontsize=20)
plt.savefig('/media/pedro/DATAPART1/AGOnIA/fraction_of_zeros_nf_recording.png')

### histograms NF, VPM, L4 ###
_,median_projection_l4,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(paths_l4[0])
_,median_projection_vpm,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(paths_vpm[0])
_,median_projection_nf,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(paths_nf[-1])
_,median_projection_nf_worst,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(paths_nf[3])
fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(10,10),sharex=True)
ax[0,0].hist(median_projection_nf.flatten(),30,label='nf')
ax[0,0].legend(fontsize=15)
ax[0,1].hist(median_projection_nf_worst.flatten(),30,label='nf_worst')
ax[0,1].legend(fontsize=15)
ax[1,0].hist(median_projection_vpm.flatten(),30,label='vpm')
ax[1,0].legend(fontsize=15)
ax[1,1].hist(median_projection_l4.flatten(),30,label='L4')
ax[1,1].legend(fontsize=15)
fig.text(0.5, 0.04, 'Fluorescence values', ha='center',fontsize=20)
plt.savefig('/media/pedro/DATAPART1/AGOnIA/fraction_of_zeros_nf_recording.svg')
plt.hist(med.flatten(),30)



range_nf_worts=[]
for im in images[:200]:
    range_nf_worts.append(max(im.flatten())-min(im.flatten()))
plt.plot(range_nf_worts)


data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(paths_nf[-1])
Yr, dims, T = cm.load_memmap(fname_new)
images_nf = np.reshape(Yr.T, [T] + list(dims), order='F')
range_nf=[]
for im in images_nf[:200]:
    range_nf.append(max(im.flatten())-min(im.flatten()))
plt.plot(range_nf)

data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(paths_vpm[0])
Yr, dims, T = cm.load_memmap(fname_new)
images_vpm = np.reshape(Yr.T, [T] + list(dims), order='F')
range_vpm=[]
for im in images_vpm[:200]:
    range_vpm.append(max(im.flatten())-min(im.flatten()))
plt.plot(range_vpm)

data_name,median_projection,fnames,fname_new,results_caiman_path,boxes_path = ut.get_files_names(paths_l4[0])
Yr, dims, T = cm.load_memmap(fname_new)
images_l4 = np.reshape(Yr.T, [T] + list(dims), order='F')
range_l4=[]
for im in images_l4[:200]:
    range_l4.append(max(im.flatten())-min(im.flatten()))
plt.plot(range_l4)

fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(10,10),sharex=True)
ax[0,0].plot(range_nf,label='nf')
ax[0,0].legend(fontsize=15)
ax[0,1].plot(range_nf_worts,label='nf_worst')
ax[0,1].legend(fontsize=15)
ax[1,0].plot(range_vpm,label='vpm')
ax[1,0].legend(fontsize=15)
ax[1,1].plot(range_l4,label='L4')
ax[1,1].legend(fontsize=15)
fig.text(0.5, 0.04, 'Frames', ha='center',fontsize=20)
fig.text(0.04, 0.5, 'Dynamic range', va='center', rotation='vertical',fontsize=20)
plt.savefig('/media/pedro/DATAPART1/AGOnIA/dynamic_range_perframe.svg')

#######################
### percentiles plots##
#######################

med_nf = np.median(images_nf,axis=0)
med_nf_worst = np.median(images_nf_worst,axis=0)
med_l4 = np.median(images_l4,axis=0)
med_vpm = np.median(images_vpm,axis=0)

nf_percentiles = [np.percentile(med_nf,x) for x in range(100)]
nf_worst_percentiles = [np.percentile(med_nf_worst,x) for x in range(100)]
l4_percentiles = [np.percentile(med_l4,x) for x in range(100)]
vpm_percentiles = [np.percentile(med_vpm,x) for x in range(100)]

fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(10,10),sharex=True)
ax[0,0].plot(nf_percentiles,label='nf')
ax[0,0].legend(fontsize=15)
ax[0,1].plot(nf_worst_percentiles,label='nf_worst')
ax[0,1].legend(fontsize=15)
ax[1,0].plot(vpm_percentiles,label='vpm')
ax[1,0].legend(fontsize=15)
ax[1,1].plot(l4_percentiles,label='L4')
ax[1,1].legend(fontsize=15)
fig.text(0.5, 0.04, 'Percentile', ha='center',fontsize=20)
fig.text(0.04, 0.5, 'Fluorescence', va='center', rotation='vertical',fontsize=20)
plt.savefig('/media/pedro/DATAPART1/AGOnIA/percentile_examples.svg')
plt.plot(gaugau)
### percentile plots for all recordings for each dataset ###
medianisima=[]
fig,axes = plt.subplots(nrows=4,ncols=2,figsize=(10,20))
for i,ax in enumerate(axes.flat[:-1]):
    recs = next(os.walk(os.path.join(PATH,data_paths[i])))[1]
    perc = []
    for recording in recs:
        _,median_projection,_,_,_,_ = ut.get_files_names(os.path.join(PATH,data_paths[i],recording))
        perc.append([np.percentile(median_projection,x) for x in range(100)])
        ax.plot(perc[-1],'gray',alpha=0.3)
    m = np.mean(perc,axis=0)
    sd = np.std(perc,axis=0)
    x = np.linspace(0,99,100)
    ax.plot(x,m,label=data_paths[i])
    ax.fill_between(x,m+sd,m-sd,alpha=0.5)
    ax.legend()
    fig.text(0.5, 0.1, 'Percentile', ha='center',fontsize=20)
    fig.text(0.04, 0.5, 'Fluorescence', va='center', rotation='vertical',fontsize=20)
    medianisima.append(m[50])
    plt.savefig('/media/pedro/DATAPART1/AGOnIA/fluorescence_percentile plots_datasets.svg')
### mean Fluorescence bellow 50th percentile for all recordings for each dataset ###
mean_bellow_50th_1 = []
for data in data_paths:
    recs = next(os.walk(os.path.join(PATH,data)))[1]
    mean_fluo = []
    for recording in recs:
        if data=='PNAS_RAW_tifs':
            try:
                median_path = os.path.join(os.path.join(PATH,data,recording),[data for data in os.listdir(os.path.join(PATH,data,recording)) if data.endswith('.jpg')
                                                  and data.startswith('MED')][0])
                if plt.imread(median_path).ndim==3:
                    median_projection = (np.array(plt.imread(median_path),dtype='uint16')*255).mean(axis=2).astype('uint16')
                else:
                    median_projection = np.array(plt.imread(median_path),dtype='uint16')*255
            except:
                print('Median projection is missing')
        else:
            _,median_projection,_,_,_,_ = ut.get_files_names(os.path.join(PATH,data,recording))
        median_projection = median_projection/np.max(median_projection)
        mean_fluo.append(np.mean(median_projection[median_projection<=np.median(median_projection)]))
    mean_bellow_50th_1.append(mean_fluo)
fig,ax = plt.subplots(figsize=(16,6))
ax.boxplot(mean_bellow_50th_1,labels=['CA1_Ch1','CA1_Ch2','L4','nf','nf_test','PNAS','VPM_corr'])
ax.tick_params(labelsize=15)
ax.set_ylabel('Mean fluorescence < Median',fontsize=20)
plt.savefig('/media/pedro/DATAPART1/AGOnIA/mean_fluorescence_bellow_median.svg')

plt.plot(np.array(medianisima)/60000,'.-')
plt.ylim([0,0.4])
for med in os.listdir('/media/pedro/Disk/PNAS medians'):
    os.makedirs(os.path.join('/media/pedro/Disk/PNAS medians',med[4:-4]))
#


































































#
