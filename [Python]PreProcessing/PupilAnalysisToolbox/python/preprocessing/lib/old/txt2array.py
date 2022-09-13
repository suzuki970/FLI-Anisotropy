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
from makeEyemetrics import makeMicroSaccade,draw_heatmap

class txt2array:    
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
    
    def dataParse(self,dat):
        
        events = {}
        for e in self.cfg["name_epoch"]:
            events[e] = []
      
        # {'Baseline':[],'.avi':[],'UE-keypress':[]}
        eyeData= {'Right':[],'Left':[]}
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
                    eyeData['Left'].append([int(line[ind_row["left"][0]]),
                                            float(line[ind_row["left"][1]]),
                                            float(line[ind_row["left"][2]]),
                                            float(line[ind_row["left"][3]])])
                    
                    eyeData['Right'].append([int(line[ind_row["right"][0]]),
                                             float(line[ind_row["right"][1]]),
                                             float(line[ind_row["right"][2]]),
                                             float(line[ind_row["right"][3]])])
            
                        
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
            
        #%% ------------- .txt to array ------------------------- %%#
        pL = np.array([p[1] for p in eyeData['Left']])
        pR = np.array([p[1] for p in eyeData['Right']])
        
        xL = np.array([p[2] for p in eyeData['Left']])
        xR = np.array([p[2] for p in eyeData['Right']])
        
        yL = np.array([p[3] for p in eyeData['Left']])
        yR = np.array([p[3] for p in eyeData['Right']])
        
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
   
        figCount = 1
        if self.cfg["visualization"]:
            plt.figure()
            # self.figSize = [3,3]
        
        dataLen = pupilData.shape[1]
        pupil_withoutInterp = pupilData.copy()
        
        #%% ------------- outlier (> 3*sigma) are replaced by zero -------------
        nBins = 201
        thre = []
        for iEyes in np.arange(pupilData.shape[0]): 
            d = np.diff(pupilData[iEyes,:])
            sigma = np.std(d)
            d = d[(d!=0) & (d>-sigma) & (d<sigma)]
            thre.append(np.std(d)*3)
        
            if self.cfg["visualization"]:
                plt.subplot(self.figSize[0],self.figSize[1],figCount)
                figCount += 1
                plt.hist(d, bins=nBins)
                plt.axvline(x=thre[iEyes], ls = "--", color='#2ca02c', alpha=0.7)
                plt.axvline(x=-thre[iEyes], ls = "--", color='#2ca02c', alpha=0.7)
        figCount += 1
        
        for iEyes in np.arange(pupilData.shape[0]): 
            ind = np.argwhere(abs(np.diff(pupilData[iEyes,:])) < thre[iEyes]).reshape(-1)
            print('Average without interp.' + self.mmName[iEyes] + ' pupil size = ' + str(np.round(pupilData[iEyes,ind].mean(),2)))
     
            ind = np.argwhere(abs(np.diff(pupilData[iEyes,])) > thre[iEyes]).reshape(-1)
            pupilData[iEyes,ind] = 0  
      
               
        eyeData["pupilData"] = pupilData
        eyeData["xData"] = xData
        eyeData["yData"] = yData
      
        return eyeData,events,self.initialTimeVal,self.fs
 
    def getAve(self,pupilData):
        # if self.cfg['normFlag']:
        tmp_p = abs(pupilData.copy())
       
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
        
        pupilData_noInterp = pupilData.copy()
        #%% ------------- blink interpolation ------------------- %%#
        pupilData = zeroInterp(pupilData.copy(),self.fs,10)
        
        zeroArray = np.zeros(eyeData["pupilData"].shape[1])
        zeroArray[pupilData["zeroArray"]] = 1
        
        
        interplatedArray = pupilData['interpolatedArray'][0]
        
        print('Interpolated array = ' + str(pupilData['interpolatedArray']) + 
              ' out of ' + str(pupilData['pupilData'].shape[1]))
        
        # if interplated array is 60% out of whole data
        if (min(np.array(pupilData['interpolatedArray']))/pupilData['pupilData'].shape[1]) > 0.5:
            rejectFlag = True
        else:
            rejectFlag = False
        
        if self.cfg['useEye'] == 1: 
            if pupilData['interpolatedArray'][0] < pupilData['interpolatedArray'][1]:
                pupilData = pupilData['pupilData'][0,:].reshape(1,pupilData['pupilData'].shape[1])
                xData = xData[0,:].reshape(-1)
                yData = yData[0,:].reshape(-1)
                useEye = 'L'
                self.mmName = ['Left']
                
            else:
                pupilData = pupilData['pupilData'][1,:].reshape(1,pupilData['pupilData'].shape[1])
                xData = xData[1,:].reshape(-1)
                yData = yData[1,:].reshape(-1)
                useEye = 'R'
                self.mmName = ['Right']
        
        elif self.cfg['useEye'] == 'L': 
            pupilData = pupilData['pupilData'][0,:].reshape(1,pupilData['pupilData'].shape[1])
            xData = xData[0,:].reshape(-1)
            yData = yData[0,:].reshape(-1)
            useEye = 'L'
            self.mmName = ['Left']
       
        elif self.cfg['useEye'] == 'R':
            pupilData = pupilData['pupilData'][1,:].reshape(1,pupilData['pupilData'].shape[1])
            xData = xData[1,:].reshape(-1)
            yData = yData[1,:].reshape(-1)
            useEye = 'R'
            self.mmName = ['Right']
        
        else: # both eyes
            pupilData = pupilData['pupilData']
            useEye = 'both'
            self.mmName = ['Left','Right']
            
        if self.cfg["useEye"] == 0: 
            for iEyes in np.arange(pupilData.shape[0]): 
                zeroInd = np.argwhere(xData[iEyes,:] == 0).reshape(-1)
                xData[1-iEyes,zeroInd] = 0
               
                zeroInd = np.argwhere(yData[iEyes,:] == 0).reshape(-1)
                yData[1-iEyes,zeroInd] = 0
                
                zeroInd = np.argwhere(pupilData[iEyes,:] == 0).reshape(-1)
                pupilData[1-iEyes,zeroInd] = 0
      
        # pupilData = np.mean(pupilData,axis=0)
        
        # xData = np.mean(xData,axis=0)
        # yData = np.mean(yData,axis=0)
        
        eyeData = {'pupilData':pupilData,
                   'zeroArray':zeroArray,
                   'gazeX':xData,
                   'gazeY':yData,
                   # 'MS':ev,
                   # 'MS_events':ms,
                   'useEye':useEye,
                   'rejectFlag':rejectFlag
                   }    
        
        return eyeData
    
    def pupilNorm(self,pupilData,ave,sigma):
        
        return (pupilData - ave) / sigma
        
 

    # def showResults(self,pupilData):
        
    #     figCount = 1
        
    #     st = 0  
    #     ed = pupilData.shape[1]
        
    #     for iEyes in np.arange(pupilData.shape[0]): 
    #         print('Average ' + self.mmName[iEyes] + ' pupil size = ' + str(np.round(pupilData[iEyes,st:ed].mean(),2)))
            
    #         plt.subplot(self.figSize[0],self.figSize[1],figCount)
    #         figCount += 1
    #         plt.plot(pupilData[iEyes,st:ed].T)
    #         plt.plot(pupil_withoutInterp[iEyes,st:ed].T,'k',alpha=0.2)
    #         # plt.ylim([5000,9000])
             
    #         plt.subplot(figSize[0],figSize[1],figCount)
    #         figCount += 1
    #         plt.plot(pupilData[iEyes,st:ed].T)
    #         plt.plot(pupil_withoutInterp[iEyes,st:ed].T,'k',alpha=0.2)
    #         # plt.ylim([5000,9000])
            
            
    #         plt.xlim([len(pupil_withoutInterp[iEyes,])/2,(len(pupil_withoutInterp[iEyes,])/2)+fs*20])
            
    #         plt.subplot(figSize[0],figSize[1],figCount)
    #         figCount += 1
    #         plt.plot(np.diff(pupilData[iEyes,st:ed]).T)
        
               
                
       
        
    #     plt.subplot(1,3,2)
    #     plt.plot(pupilData.T)
    #     plt.xlim([200000, 210000])
    #     # plt.ylim([20000,10000])
        
    #     plt.subplot(1,3,3)
    #     plt.plot(np.diff(pupilData).T)
    #     plt.xlim([200000, 210000])
    #     plt.ylim([-50,50])
       
