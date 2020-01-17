import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import skimage.io as io
from skimage.measure import points_in_poly as isInROI
from skimage.measure import find_contours as FC
import math
import itertools
import pandas as pd
from tifffile import imsave
import json
import zipfile
import struct
import tempfile 
import glob
import shutil

def pixwiseRunningBSL(activityTrace, FPS = 3, smoothingWindow = 30, percentile = 20):
    '''
       Input: pixel-wise fluorescence intensity trace as a np.array (or slice of it) 
       Output: t-series of (F-f0)/fo and f0
       As default f0 is computed as the n-th percentile (default 20) of fluorescence intensity along a user defined time window (in
       seconds;default is 30). 
       
       This function computes the n-th percentile using the NEXT *smoothingWindow* seconds.
       
       To allow for f0 normalization if f0 goes to 0 it's substituted by its actual value in the input trace.
       Outputs:
       smoothedTrace and bslTrace which are both np.arrays
       Created by Zebastiano
       '''
    frameNumber = activityTrace.shape[0]
    
    bslTrace = np.zeros(frameNumber)
    smoothedTrace = np.zeros(frameNumber)
    
    window = round(smoothingWindow*FPS)
    
    for i in np.arange(0,frameNumber-window):
        bslTrace[i]=(np.percentile(a = activityTrace[i:(i+window)], q = percentile, interpolation = 'linear'))
    for i in np.arange(frameNumber-window,frameNumber):
        bslTrace[i]=(np.percentile(a = activityTrace[i:frameNumber], q = percentile, interpolation = 'linear'))
    
    zeros = np.where(bslTrace == 0)
    bslTrace[zeros] = 1
    smoothedTrace = (activityTrace - bslTrace)/bslTrace
        
    return(smoothedTrace, bslTrace)


def centered_pixwiseRunningBSL(activityTrace, FPS = 3, smoothingWindow = 30, percentile = 20):
    '''
       Input: pixel-wise fluorescence intensity trace as a np.array (or slice of it) 
       Output: t-series of (F-f0)/fo and f0
       As default f0 is computed as the n-th percentile (default 20) of fluorescence intensity along a user defined time window (in
       seconds;default is 30). 
       
       This function computes the n-th percentile CENTERED on a *smoothingWindow* [s] wide window.
       
       To allow for f0 normalization if f0 goes to 0 it's substituted by its actual value in the input trace.
       Outputs:
       smoothedTrace and bslTrace which are both np.arrays
       Created by Zebastiano
       '''
    frameNumber = activityTrace.shape[0]
    
    bslTrace = np.zeros(frameNumber)
    smoothedTrace = np.zeros(frameNumber)
    
    window = round(smoothingWindow*FPS)
    window_half1 = int(math.trunc(window/2))
    window_half2 = int(math.ceil(window/2))
    
    if(window_half1+window_half2)==window:
        for i in np.arange(0 , window_half1):
            bslTrace[i]=(np.percentile(a = activityTrace[window_half1-(window_half1-i):(i+window_half1)], q = percentile, interpolation = 'linear'))
        for i in np.arange(window_half1,frameNumber-window_half2):
            bslTrace[i]=(np.percentile(a = activityTrace[(i-window_half1):(i+window_half2)], q = percentile, interpolation = 'linear'))
        for i in np.arange(frameNumber-window_half2,frameNumber):
            bslTrace[i]=(np.percentile(a = activityTrace[(i-window_half1):(frameNumber-(window_half2-i))], q = percentile, interpolation = 'linear'))
    
        zeros = np.where(bslTrace == 0)
        bslTrace[zeros] = 1
        smoothedTrace = (activityTrace - bslTrace)/bslTrace
    else:
        print('error')
    return(smoothedTrace, bslTrace)

def evs(trx, evsLimits):
    '''
    Input: 
    *trx* as np.array 
    event limits (*evsLimits*) as df.
    Returns: 
    *events_trace* a filtered trace containg just the significative events
    *events_list* a df containing all the events in the columns, indexes are frames

    Note:
    This function is cool because it uses itertools to append a list of lists of different lenght to df columns
    ''' 
    events_trace = np.zeros(trx.shape[0])

    events_columns = ['Event_'+str(i) for i in np.arange(0,evsLimits.shape[0])]
    evsDF = pd.DataFrame(columns = events_columns)

    events_list = []

    for i in evsLimits.index:
        start = evsLimits.at[i,'event_start']
        stop = evsLimits.at[i,'event_stop']
        tempTrace = trx[start:stop]
        events_trace[start:stop] = tempTrace
        tempTrace = tempTrace.tolist()
        events_list.append(tempTrace)

    eventDF = pd.DataFrame((_ for _ in itertools.zip_longest(*events_list)), columns=events_columns)
    return(events_trace, eventDF)

def eventFinder(trace = None, start_nSigma_bsl = 2, stop_nSigma_bsl = .5, FPS = 3, minimumDuration = .5, debugPlot = False):
    ''' This function gets a *trace* as input, computes the whole trace SD and filter the trace. 
    The filtered trace will be depleted of the biggest events, and considered like the baseline. A new SD (*bslSD*) is computed
    and start and stop thresholds will be computed as n times *bslSD* according to *start_nSigma_bsl* (default = 2), *stop_nSigma_bsl* (default = .5).
    
    Events (pos or neg) are defined as *trace* segments above startTreshold and stopTreshold and duration bigger than *minimumDuration* in [s].
    FPS has to be passed in order to compute the duration. In this section events are identified as transition points through the tresholds (limits).
    
    Then calling the ad hoc fucntion *evs* computes the *event_trace* and the *event_df* for both positive and negative events.
    
    Returns:
    
    pos_events_trace
    neg_events_trace
    positiveEventsDF_limits --> [in frames]
    negativeEventsDF_limits --> [in frames]
    pos_eventDF --> all positve events ordered according appearence
    neg_eventDF --> all negative events ordered according appearence
    
    Note:
    The ratio of (neg_eventDF/pos_eventDF) represent a measure of false discovery rate see Dombeck DA 2007.
    '''
    SD = trace.std()
    belowSD = np.where(trace<trace.std(), trace, np.nan)
    
    belowSD_nanRemoved = belowSD[~np.isnan(belowSD)]
    bslSD = belowSD_nanRemoved.std()
    
    startTreshold = start_nSigma_bsl*bslSD
    stopTreshold = stop_nSigma_bsl*bslSD
    
        
    pos_eventStart = []
    pos_eventStop = []
    neg_eventStart = []
    neg_eventStop = []
    #for i in np.arange(0,centered_smoothedTrace.shape[0]):
    i=0
    while i<trace.shape[0]-1:
            
        while trace[i]>=startTreshold and i<(trace.shape[0]-2):
            pos_eventStart.append(i)
            while trace[i]>=stopTreshold and i<(trace.shape[0]-2):
                i = i+1
            pos_eventStop.append(i)
            i = i+1
            
        while trace[i]<=-startTreshold and (i<trace.shape[0]-2):
            neg_eventStart.append(i)
            while trace[i]<=-stopTreshold and (i<trace.shape[0]-2):
                i = i+1
            neg_eventStop.append(i)
            i = i+1
        #if i<(trace.shape[0]-2):
        i = i+1
    
    if len(pos_eventStart) != 0:
        if len(pos_eventStart)/len(pos_eventStart) != 1:
            if len(pos_eventStart) > len(pos_eventStop):
                posToRemove = len(pos_eventStart) - len(pos_eventStop)
                pos_eventStart = pos_eventStart[:len(pos_eventStart)-posToRemove]
            if len(pos_eventStop) > len(pos_eventStart):
                posToRemove = len(pos_eventStop) - len(pos_eventStart)
                pos_eventStop = pos_eventStop[:len(pos_eventStop)-posToRemove]

    if len(neg_eventStop) != 0:
        if len(neg_eventStart)/len(neg_eventStop) != 1:
            if len(neg_eventStart) > len(neg_eventStop):
                posToRemove = len(neg_eventStart) - len(neg_eventStop)
                neg_eventStart = neg_eventStart[:len(neg_eventStart)-posToRemove]
            if len(neg_eventStop) > len(neg_eventStart):
                posToRemove = len(neg_eventStop) - len(neg_eventStart)
                neg_eventStop = neg_eventStop[:len(neg_eventStop)-posToRemove]

    data_pos = [pos_eventStart,pos_eventStop]
    positiveEventsDF = pd.DataFrame(data = data_pos, index=['event_start','event_stop'])
    positiveEventsDF = positiveEventsDF.transpose()
    positiveEventsDF['event_duration'] = (positiveEventsDF.event_stop - positiveEventsDF.event_start)/FPS 
    positiveEventsDF_limits = positiveEventsDF[positiveEventsDF.event_duration >= minimumDuration]
    positiveEventsDF_limits.reset_index(drop=True, inplace = True)
    
    data_neg = [neg_eventStart,neg_eventStop]
    negativeEventsDF = pd.DataFrame(data = data_neg, index=['event_start','event_stop'])
    negativeEventsDF = negativeEventsDF.transpose()
    negativeEventsDF['event_duration'] = (negativeEventsDF.event_stop - negativeEventsDF.event_start)/FPS 
    negativeEventsDF_limits = negativeEventsDF[negativeEventsDF.event_duration >= minimumDuration]
    negativeEventsDF_limits.reset_index(drop=True, inplace = True)
    
    
    
    
    pos_events_trace, pos_eventDF = evs(trx = trace, evsLimits = positiveEventsDF_limits)
    neg_events_trace, neg_eventDF = evs(trx = trace, evsLimits = negativeEventsDF_limits)
        
    if debugPlot == True:
        ax = plt.subplot(311)
        plt.plot(np.arange(trace.shape[0]), trace, 'gray')
        ax.axhline(y = SD, c = 'b')
        plt.plot(np.arange(belowSD.shape[0]), belowSD, 'b')
        ax.axhline(y = start_nSigma_bsl*bslSD, c = 'r', ls = ':', lw = .3)
        ax.axhline(y = -start_nSigma_bsl*bslSD, c = 'k', ls = ':', lw = .3)
        ax.axhline(y = stop_nSigma_bsl*bslSD, c = 'r',  ls = ':', lw = .3)
        ax.axhline(y = -stop_nSigma_bsl*bslSD, c = 'k',  ls = ':', lw = .3)

        ax2 = plt.subplot(312)
        plt.plot(np.arange(trace.shape[0]), trace, 'gray')
        plt.plot(pos_eventStart, trace[pos_eventStart], '.r')
        plt.plot(pos_eventStop, trace[pos_eventStop], '.k')

        plt.plot(neg_eventStart, trace[neg_eventStart], '.g')
        plt.plot(neg_eventStop, trace[neg_eventStop], '.y')
        
        ax3 = plt.subplot(313)
        plt.plot(np.arange(trace.shape[0]),trace, 'gray')
        plt.plot(np.arange(trace.shape[0]),pos_events_trace, c = 'r', alpha = .7)
        plt.plot(np.arange(trace.shape[0]),neg_events_trace, c = 'b', alpha = .7)
        plt.show()
    
    return(pos_events_trace, neg_events_trace, positiveEventsDF_limits, negativeEventsDF_limits, pos_eventDF, neg_eventDF)