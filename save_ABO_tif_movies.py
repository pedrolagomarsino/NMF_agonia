import os
import tifffile
import h5py as h
import numpy as np
import matplotlib.pyplot as plt

PATH = '/media/pedro/DATA_BUCKET/PNAS_RAW/'
SAVE_PATH = '/media/pedro/DATA_BUCKET/PNAS_RAW_tifs'
for exp in [x for x in os.listdir(PATH) if x.endswith('.h5')]:
    f= h.File(os.path.join(PATH,exp),'r')
    data = f['data']
    with tifffile.TiffWriter(os.path.join(SAVE_PATH,os.path.splitext(exp)[0]+'.tif'), imagej=True) as tif:
        tif.save(data[:5000], metadata={'axes':'TXY'})
