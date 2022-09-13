import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zeroInterp import zeroInterp
from pre_processing import pre_processing

mat = scipy.io.loadmat('dataForPython.mat')
allPupilData = mat['PLR']

cfg={
'SAMPLING_RATE':250,   
'windowL':10,
'TIME_START':-1,
'TIME_END':4,
'WID_ANALYSIS':4,
'WID_BASELINE':np.array([-0.2,0]),
'WID_FILTER':np.array([]),
'METHOD':1, #subtraction
'FLAG_LOWPASS':False,
'THRES_DIFF':0.2
# 'THRES_DIFF':300
}

startTime = -1
endTime = 4

x = np.arange(startTime,endTime,((endTime - startTime) / (allPupilData.shape[1])))

plt.figure(figsize=(15.0, 3.0))
plt.subplot(1,3,1)
plt.plot(x,allPupilData.T)
plt.title('raw data',fontsize=14)
plt.xlabel('time[s]',fontsize=14)
plt.ylabel('pupil diameter [mm]',fontsize=14)

## blink interpolation
y = zeroInterp(allPupilData.copy(),cfg['SAMPLING_RATE'],cfg['windowL'])
y = y['pupilData']

#y = y[np.arange(0,1),]
plt.subplot(1,3,2)
plt.plot(x,y.T)
plt.ylim([0,6])
plt.title('interpolated data',fontsize=14)
plt.xlabel('time[s]',fontsize=14)
plt.ylabel('pupil diameter [mm]',fontsize=14)
   
### pre-processing
y,rejectNum = pre_processing(y,cfg)

y = np.delete(y,rejectNum,axis=0)
x = np.arange(startTime,endTime,((endTime - startTime) / (y.shape[1])))

plt.subplot(1,3,3)
plt.plot(x,y.T)
plt.title('baseline-corrected data',fontsize=14)
plt.xlabel('time[s]',fontsize=14)
plt.ylabel('pupil diameter [mm]',fontsize=14)
