"""
Input:    
dat      - list of asc file transfered from EDF Converter 
              provided from SR-research(https://www.sr-support.com/thread-23.html)
cfg      - dict of parameters for analysis

Output:
eyeData   - dict which includes pupil, gaze and micro-saccades
events    - list of triggers in asc file
initialTimeVal - recordig start timing
fs        - sampling rate of the recording

Example:
    fileName = glob.glob(os.path.join('/xxx.asc'))
    f = open(os.path.join(str(fileName[0])))
    dat=[]
    for line in f.readlines():
        dat.append(line.split())
    f.close()
    cfg={'useEye':2,
        'WID_FILTER':[],
        'mmFlag':False,
        'normFlag':True,
        's_trg':'SYNCTIME',
        'visualization':False
        }
    eyeData,events,initialTimeVal,fs = asc2array(dat, cfg)
    
    pupilData = eyeData['pupilData']
    gazeX = eyeData['gazeX']
    gazeY = eyeData['gazeY']
    mSaccade = eyeData['mSaccade']
"""

import numpy as np
from zeroInterp import zeroInterp
import matplotlib.pyplot as plt
from pre_processing import re_sampling
from band_pass_filter import butter_bandpass_filter,lowpass_filter
from au2mm import au2mm
from makeEyemetrics import makeMicroSaccade,draw_heatmap

class asc2array_cls:    
    def __init__(self, cfg):
        self.cfg = cfg
    
        if len(cfg['s_trg']) > 0:
            self.s_trg = cfg['s_trg']
        else:
            self.s_trg = 'Start_Experiment'

        if self.cfg["visualization"]:
            self.figSize = [3,3]

        self.mmName = ['Left','Right']

    def dataExtraction(self,fileName):
        
        f = open(fileName)
        dat=[]
        for line in f.readlines():
            dat.append(line.split())
            
        f.close()
        
        
        
        
        return dat
    
    def dataParse(self, dat, EyeTracker = "Eyelink"):
        
        print('---------------------------------')
        print('Analysing...')
  
        eyeData= {'Right':[],'Left':[]}

        if EyeTracker == "Eyelink":
            events = {'SFIX':[],'EFIX':[],'SSACC':[],'ESACC':[],'SBLINK':[],'EBLINK':[],'MSG':[]}
            
            msg_type = ['SFIX','EFIX','SSACC','ESACC','SBLINK','EBLINK','MSG']
            
            start = False
            for line in dat:
                if start:
                    if len(line) > 3:
                        if line[0].isdecimal() and line[1].replace('.','',1).isdigit() :
                            eyeData['Left'].append([float(line[0]),
                                                    float(line[1]),
                                                    float(line[2]),
                                                    float(line[3])])
                        if line[0].isdecimal() and line[4].replace('.','',1).isdigit() :
                            eyeData['Right'].append([float(line[0]),
                                                      float(line[4]),
                                                      float(line[5]),
                                                      float(line[6])])
                        
                    for m in msg_type:
                        if line[0] == m:
                              events[m].append(line[1:])
                else:
                    if 'RATE' in line:
                        self.fs = float(line[5])
                    if self.s_trg in line:
                        start = True
                        self.initialTimeVal = int(line[1])
                        if self.s_trg != 'Start_Experiment':
                            events['MSG'].append(line[1:])
        
        elif EyeTracker == "SMI":
            events = {}
            for e in self.cfg["name_epoch"]:
                events[e] = []
          
            ind_row = {"left":[],"right":[]}
         
            start = False
            for line in dat:
                if start:
                    if 'SMP' in line:
                            
                        # if float(line[ind_row["left"][2]]) == 0: # if gaze position couldn't detect, pupil position adopts instead
                        #     eyeData['Left'].append([int(line[ind_row["left"][0]]),
                        #                             float(line[ind_row["left"][1]]),
                        #                             float(line[ind_row["left"][4]]),
                        #                             float(line[ind_row["left"][5]])])
                            
                        #     eyeData['Right'].append([int(line[ind_row["right"][0]]),
                        #                             float(line[ind_row["right"][1]]),
                        #                             float(line[ind_row["right"][4]]),
                        #                             float(line[ind_row["right"][5]])])
                        #     # print("warning!")
                        # else:
                        eyeData['Left'].append([int(line[ind_row["left"][0]]),    # time stamp
                                                float(line[ind_row["left"][2]]),  # gaze x
                                                float(line[ind_row["left"][3]]),  # gaze y
                                                float(line[ind_row["left"][1]])]) # pupil
                        
                        eyeData['Right'].append([int(line[ind_row["right"][0]]),
                                                 float(line[ind_row["right"][2]]),
                                                 float(line[ind_row["right"][3]]),
                                                 float(line[ind_row["right"][1]])])
                
                            
                    for m in self.cfg["name_epoch"]:
                        if m in str(line[5]):
                             events[m].append([int(line[0]),line[5:]])
                      
                else:
                    
                    if line[0]=="Time":
                        row = []
                        for i in np.arange(len(line)-2):
                            if line[i] == "Time" or line[i] == "Type" or line[i] == "Trial":
                                row.append(line[i])
                            if line[i] == "L" or line[i] == "R":
                                row.append(line[i]+"_"+line[i+1]+"_"+line[i+2])
                      
                        for i,r in enumerate(row):
                            if r == "Time" or r == "L_Mapped_Diameter" or r == "L_POR_X" or r == "L_POR_Y":
                                ind_row["left"].append(i)
                            if r == "Time" or r == "R_Mapped_Diameter" or r == "R_POR_X" or r == "R_POR_Y" :
                                ind_row["right"].append(i)
                        
                        for i,r in enumerate(row): # pupil position
                            if r == "L_Raw_X" or r == "L_Raw_Y":
                                ind_row["left"].append(i)
                            
                            if r == "R_Raw_X" or r == "R_Raw_Y":
                                ind_row["right"].append(i)
                            
                    if 'Rate:' in line:
                        self.fs = int(line[3])
                        
                    if self.cfg['s_trg'] in line:
                        start = True
                        events[self.cfg["name_epoch"][0]].append([int(line[0]),line[5:]])
                        self.initialTimeVal = int(line[0])
                        # if s_trg != 'Start_Experiment':
                        #     events[s_trg].append(line[1:])
                        
                        
        #%% ------------- .asc to array ------------- %%#
        xL = np.array([p[1] for p in eyeData['Left']])
        xR = np.array([p[1] for p in eyeData['Right']])
        
        yL = np.array([p[2] for p in eyeData['Left']])
        yR = np.array([p[2] for p in eyeData['Right']])
        
        pL = np.array([p[3] for p in eyeData['Left']])
        pR = np.array([p[3] for p in eyeData['Right']])
        
        if EyeTracker == "Eyelink":
            timeStampL = np.array([p[0] for p in eyeData['Left']])
            timeStampR = np.array([p[0] for p in eyeData['Right']])
         
            timeStampL = [int((t - self.initialTimeVal)*(self.fs/1000)) for t in timeStampL]
            timeStampR = [int((t - self.initialTimeVal)*(self.fs/1000)) for t in timeStampR]
        
            # for m in list(events.keys()):
            # events["MSG"] = [[int(((int(p[0])- self.initialTimeVal) * (self.fs / 10**6))),p[1:]] for p in events["MSG"]]
        
        elif EyeTracker == "SMI":         
            timeStampL = np.array([int(((p[0]- self.initialTimeVal)/1000) / (1 / self.fs * 1000)) for p in eyeData['Left']])
            timeStampR = np.array([int(((p[0]- self.initialTimeVal)/1000) / (1 / self.fs * 1000)) for p in eyeData['Right']])
       
        timeLen = np.max(timeStampR) if np.max(timeStampR) > np.max(timeStampL) else np.max(timeStampL)
    
        pupilData = np.zeros((2,timeLen+1))
        pupilData[0,timeStampL] = pL
        pupilData[1,timeStampR] = pR
        
        xData = np.zeros((2,timeLen+1))
        xData[0,timeStampL] = xL
        xData[1,timeStampR] = xR
        
        yData = np.zeros((2,timeLen+1))
        yData[0,timeStampL] = yL
        yData[1,timeStampR] = yR
         
        interpolatedArray = []
        interpolatedArray.append(np.argwhere(pupilData[0,]==0).reshape(-1))
        interpolatedArray.append(np.argwhere(pupilData[1,]==0).reshape(-1))
            
        #%% ------------- resampling -------------
        figCount = 1
        if self.cfg["visualization"]:
            plt.figure()
            figSize = [3,3]
            
        self.dataLen = pupilData.shape[1]
        pupil_withoutInterp = pupilData.copy()
        
        if self.fs > 250: # temporary down-sampled to 250Hz for blink interpolation
            pupilData = re_sampling(pupilData.copy(),round(self.dataLen*( 250 / self.fs)))
        else:
            pupilData = pupilData
           
        #%% ------------- outlier (> 3*sigma) are replaced by zero -------------
        nBins = 201
        thre = []
        for iEyes in np.arange(pupilData.shape[0]): 
            d = np.diff(pupilData[iEyes,:])
            sigma = np.std(d)
            d = d[(d!=0) & (d > -sigma) & (d < sigma)]
            thre.append(np.std(d)*3)
        
            if self.cfg["visualization"]:
                plt.subplot(figSize[0],figSize[1],figCount)
                figCount += 1
                plt.hist(d, bins=nBins)
                plt.axvline(x=thre[iEyes], ls = "--", color='#2ca02c', alpha=0.7)
                plt.axvline(x=-thre[iEyes], ls = "--", color='#2ca02c', alpha=0.7)
        figCount += 1
          
        for iEyes in np.arange(pupilData.shape[0]): 
            ind = np.argwhere(abs(np.diff(pupilData[iEyes,:])) < thre[iEyes]).reshape(-1)
            # print('Average without interp.' + mmName[iEyes] + ' pupil size = ' + str(np.round(pupilData[iEyes,ind].mean(),2)))
     
            ind = np.argwhere(abs(np.diff(pupilData[iEyes,])) > thre[iEyes]).reshape(-1)
            pupilData[iEyes,ind] = 0  
    
        eyeData["pupilData"] = pupilData
        eyeData["xData"] = xData
        eyeData["yData"] = yData
        eyeData["interpolatedArray"] = interpolatedArray
      
        return eyeData,events,self.initialTimeVal,self.fs
    
    def getAve(self,pupilData):
        tmp_p = abs(pupilData.copy())
        
        if self.cfg["mmFlag"]:
            tmp_p = (tmp_p/256)**2*np.pi
            tmp_p = np.sqrt(tmp_p) * au2mm(700) 
        
        ind_nonzero = np.argwhere(tmp_p[0,:] != 0).reshape(-1)
        ave_left = np.mean(tmp_p[0,ind_nonzero])
        sigma_left = np.std(tmp_p[0,ind_nonzero])   
        
        ind_nonzero = np.argwhere(tmp_p[1,:] != 0).reshape(-1)
        ave_right = np.mean(tmp_p[1,ind_nonzero])
        sigma_right = np.std(tmp_p[1,ind_nonzero])       
        
        ave = np.mean([ave_left,ave_right])
        sigma = np.mean([sigma_left,sigma_right])
        
        print('Normalized by mu = ' + str(np.round(ave,4)) + ' sigma = ' + str(np.round(sigma,4)))
     
        return ave,sigma
    
    def blinkInterp(self,eyeData):
        
        pupilData = eyeData["pupilData"]
        xData = eyeData["xData"]
        yData = eyeData["yData"]
                
        if self.fs > 250: # temporary down-sampled to 250Hz for blink interpolation
            pupilData = zeroInterp(pupilData.copy(),250,10)
        else:
            pupilData = zeroInterp(pupilData.copy(),self.fs,10)
            # xData = zeroInterp(xData.copy(),self.fs,10)['pupilData']
            # yData = zeroInterp(yData.copy(),self.fs,10)['pupilData']

        zeroArray = np.zeros(eyeData["pupilData"].shape[1])
        zeroArray[pupilData["zeroArray"]] = 1
        # data_test = pupilData['data_test']
        # data_control_bef = pupilData['data_control_bef']
        # data_control_aft = pupilData['data_control_aft']
        # interpolatedArray = pupilData['zeroArray']
        
        print('Interpolated array = ' + str(pupilData['interpolatedArray']) + 
              ' out of ' + str(pupilData['pupilData'].shape[1]))
        
        # if interplated array is 60% out of whole data
        if (min(np.array(pupilData['interpolatedArray']))/pupilData['pupilData'].shape[1]) > 0.4:
            rejectFlag = True
        else:
            rejectFlag = False
            
           
        if self.cfg['useEye'] == 1: 
            if pupilData['interpolatedArray'][0] < pupilData['interpolatedArray'][1]:
                pupilData = pupilData['pupilData'][0,:].reshape(1,pupilData['pupilData'].shape[1])
                xData = xData[0,:].reshape(1,xData.shape[1])
                yData = yData[0,:].reshape(1,yData.shape[1])
                useEye = 'L'
                mmName = ['Left']
                 
            else:
                pupilData = pupilData['pupilData'][1,:].reshape(1,pupilData['pupilData'].shape[1])
                xData = xData[1,:].reshape(1,xData.shape[1])
                yData = yData[1,:].reshape(1,yData.shape[1])
                useEye = 'R'
                mmName = ['Right']
                
        elif self.cfg['useEye'] == 'L': 
            pupilData = pupilData['pupilData'][0,:].reshape(1,pupilData['pupilData'].shape[1])
            xData = xData[0,:].reshape(1,xData.shape[1])
            yData = yData[0,:].reshape(1,yData.shape[1])
            useEye = 'L'
            mmName = ['Left']
            
        elif self.cfg['useEye'] == 'R':
            pupilData = pupilData['pupilData'][1,:].reshape(1,pupilData['pupilData'].shape[1])
            xData = xData[1,:].reshape(1,xData.shape[1])
            yData = yData[1,:].reshape(1,yData.shape[1])
            useEye = 'R'
            mmName = ['Right']
       
        else: # both eyes
            pupilData = pupilData['pupilData']
            useEye = 'both'
            mmName = ['Left','Right']
   
        if self.cfg["useEye"] == 0: 
            for iEyes in np.arange(pupilData.shape[0]): 
                zeroInd = np.argwhere(xData[iEyes,:] == 0).reshape(-1)
                xData[1-iEyes,zeroInd] = 0
               
                zeroInd = np.argwhere(yData[iEyes,:] == 0).reshape(-1)
                yData[1-iEyes,zeroInd] = 0
                
                zeroInd = np.argwhere(pupilData[iEyes,:] == 0).reshape(-1)
                pupilData[1-iEyes,zeroInd] = 0
      
        # xData = re_sampling(xData.copy(),self.dataLen)
        # yData = re_sampling(yData.copy(),self.dataLen)
        pupilData = re_sampling(pupilData.copy(),self.dataLen)

        
        if self.cfg["mmFlag"]:
            pupilData = abs(pupilData)
            pupilData = (pupilData/256)**2*np.pi
            pupilData = np.sqrt(pupilData) * au2mm(700)
            # pupilData = 1.7*(10**(-4))*480*np.sqrt(pupilData)
    
    
        #%% -------------  data save ------------- %%#
        eyeData = {'pupilData':pupilData,
                'gazeX':xData,
                'gazeY':yData,
                "zeroArray":zeroArray,
                # 'MS':ev,
                # 'MS_events':ms,
                'useEye':useEye,
                'rejectFlag':rejectFlag,
                # 'data_test':data_test,
                # 'data_control_bef':data_control_bef,
                # 'data_control_aft':data_control_aft,
                'interpolatedArray':eyeData["interpolatedArray"]
                }
        
        return eyeData
#     pupilData = re_sampling(pupilData.copy(),dataLen)

    def pupilNorm(self,pupilData,ave,sigma):
        
        return (pupilData - ave) / sigma
        
    def showResults(self,events,eyeData):
        
        pupilData = eyeData["pupilData"]
        #%% -------------  show results ------------- %%#
     
        st = [[int(int(e[0])- self.initialTimeVal),e[1]] for e in events['MSG'] if 'Start' in e[1]]       
        ed = [[int(int(e[0])- self.initialTimeVal),e[1]] for e in events['MSG'] if 'End' in e[1]]       
        
        if (len(st) == 0) | (len(ed) == 0):
            st.append([0])
            ed.append([pupilData.shape[1]])
        
        for iEyes in np.arange(pupilData.shape[0]): 
            print('Average ' + self.mmName[iEyes] + ' pupil size = ' + str(np.round(pupilData[iEyes,st[0][0]:ed[0][0]].mean(),2)))
            
            # if self.cfg["visualization"]:
            #     plt.subplot(figSize[0],figSize[1],figCount)
            #     figCount += 1
            #     plt.plot(pupilData[iEyes,st[0][0]:ed[0][0]].T)
            #     plt.plot(pupil_withoutInterp[iEyes,st[0][0]:ed[0][0]].T,'k',alpha=0.2)
            #     plt.ylim([5000,9000])
                 
            #     plt.subplot(figSize[0],figSize[1],figCount)
            #     figCount += 1
            #     plt.plot(pupilData[iEyes,st[0][0]:ed[0][0]].T)
            #     plt.plot(pupil_withoutInterp[iEyes,st[0][0]:ed[0][0]].T,'k',alpha=0.2)
            #     plt.ylim([5000,9000])
            #     plt.xlim([45000,50000])
                
            #     plt.subplot(figSize[0],figSize[1],figCount)
            #     figCount += 1
            #     plt.plot(np.diff(pupilData[iEyes,st[0][0]:ed[0][0]]).T)
                
                # plt.hlines(upsilon, 0, len(xData), "red", linestyles='dashed')
                # plt.savefig("./img.pdf")
        # print('upsilon = ' + str(np.round(upsilon,4)) + ', std = ' + str(np.round(np.nanstd(v),4)))

    #%% -------------  data plot ------------- %%#
    
    # pupilData = np.mean(pupilData,axis=0)
    # xData = np.mean(xData,axis=0)
    # yData = np.mean(yData,axis=0)
    
    
    # plt.plot(pupilData.T,color="k")
    # plt.ylim([0,10000])
    
    # plt.subplot(2,3,2)
    # plt.plot(np.diff(pupilData).T,color="k")
    # plt.ylim([-50,50])
    
    # plt.subplot(1,3,2)
    # plt.plot(pupilData.T)
    # plt.xlim([200000, 210000])
    # # plt.ylim([20000,10000])
    
    # plt.subplot(1,3,3)
    # plt.plot(np.diff(pupilData).T)
    # plt.xlim([200000, 210000])
    # plt.ylim([-50,50])
    # plt.subplot(2,3,4)
    # plt.plot(pupilData.T,color="k")
    # plt.xlim([500000, 550000])
    # plt.ylim([0,10000])
    
    # plt.subplot(2,3,5)
    # plt.plot(pupilData.T,color="k")
    # plt.xlim([1000000, 1050000])
    # plt.ylim([0,10000])
    
    # plt.subplot(2,3,6)
    # plt.plot(pupilData.T,color="k")
    # plt.xlim([2000000, 2050000])
    # plt.ylim([0,10000])

    # if normFlag:
    #     pupilData = (pupilData - ave) / sigma
              
    # if len(filt) > 0:
    #     pupilData = butter_bandpass_filter(pupilData, filt[0], filt[1], fs, order=4)

    # return eyeData,events,self.initialTimeVal,int(fs)
