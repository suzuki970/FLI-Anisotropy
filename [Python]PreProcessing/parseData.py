import numpy as np
import json
from asc2array_cls import asc2array_cls
import glob
import os
import itertools
import datetime
from makeFixation import fixation_detection
from pre_processing_cls import getNearestValue

def inclusive_index(lst, purpose):
    ind = []
    for i, e in enumerate(lst):
        if purpose in e: 
            ind.append(i)
            
    return ind
 
name_epoch = [
   "Baseline",
   ".avi",
   "UE-keypress"]
  
name_condition = [
    "Lag-83",
    "Lag-67",
    "Lag-50",
    "Lag-33",
    "Lag-17",
    "Lag0",
    "Lag17",
    "Lag33"]

cfg={"TIME_START":-2,
     "TIME_END":3.55,
     # "THRES_DIFF":10,
     "WID_ANALYSIS":5.55,
     "WID_BP_ANALYSIS":1,
     "useEye":0,
     "name_epoch":name_epoch,
     "WID_FILTER":[],
     # "mmFlag":True,
     "mmFlag":False,
     "normFlag":True,
       # "normFlag":False,
     "s_trg":"001Baseline.jpg",
     "visualization":False,
     "MS":False,
     "RESAMPLING_RATE":60,
     "DOT_PITCH":0.282,   
     "VISUAL_DISTANCE":70,
     "acceptMSRange":4.5, # farest locs of the stim
     "MIN_DURATION":0.1,
     "SCREEN_RES":[1680, 1050],
     "SCREEN_MM":[473.76, 296.1]
     }

  
if not cfg['mmFlag'] and not cfg['normFlag']:
    unitName = "_mm"
elif cfg['mmFlag']:
    unitName = "_mm"
else:
    unitName = "_norm" 


t2a = asc2array_cls(cfg) # make instance

NumOfconditions = len(name_condition)
useEyeFlg = []

dt_now = datetime.datetime.now()
date = dt_now.strftime("%Y%m%d")

#%%
for sideRoop in ["LVF","RVF"]:  
    
    rootFolder = "./results/"+sideRoop+"/"
   
    cfg["rejectFlag"] = []
    
    folderList=[]
    
    for filename in os.listdir(rootFolder):
        if os.path.isdir(os.path.join(rootFolder, filename)): 
            folderList.append(filename)
    folderList.sort()

    datHash={"PDR":[],
             "PDR_res":[],
             "zeroArray":[],
             "allAvePDR":[],
             "baselinePD":[],
             "gazeX":[],
             "gazeY":[],
             "SOA":[],
             "RT":[],
             "Task":[],
             "sub":[],
             "rejectFlag":[]
              }
    
    for iSub,subName in enumerate(folderList):
       
        print("Processing --> " + subName[-3:] + "...")
        
        fileName = glob.glob(os.path.join(rootFolder+subName+"/*Samples.txt"))
    
        #%% ------------------ load eye-data file -------------------
        dat = t2a.dataExtraction(fileName[0])
        
        #%% ------------------ parse data to variable -------------------
        eyeData,events,initialTimeVal,fs = t2a.dataParse(dat,"SMI")
        ave,sigma = t2a.getAve(eyeData["pupilData"])

        #%% ------------------ blink interpolation -------------------
        
        eyeData = t2a.blinkInterp(eyeData)
        
        if cfg["normFlag"]:
            pupilData = t2a.pupilNorm(eyeData["pupilData"], ave, sigma)
        else:
            pupilData = eyeData["pupilData"]
        
        pupilData = np.mean(pupilData,axis=0)
        
        gazeX = np.mean(eyeData["gazeX"],axis=0)
        gazeY = np.mean(eyeData["gazeY"],axis=0)
        
        zeroArray = eyeData["zeroArray"]
        
        cfg["rejectFlag"].append(eyeData["rejectFlag"])
   
        for m in name_epoch:
            events[m] = [[int(((p[0]- initialTimeVal)/1000) / (1 / fs * 1000)),p[1]]
                              for p in events[m]]    
       
        #%% ------------------ load response file -------------------
        
        fileName = glob.glob(os.path.join(rootFolder+subName+"/*protocol.txt"))
        if len(fileName) > 0:
            f = open(os.path.join(str(fileName[0])))
            dat_ev={".avi":[],
                    "UE-keypress":[],
                    "Baseline":[]
                    }
            
            start = False
            for line in f.readlines():
                if start:
                    if ("ET_REM" in line) or ("ET_CNT" in line):
                        if "Baseline" in line.split()[2]:
                            dat_ev["Baseline"].append([int(line.split()[0]),line.split()[2:]])
                        if ".avi" in line.split()[2]:
                            dat_ev[".avi"].append([int(line.split()[0]),line.split()[2:]])
                        if "UE-keypress" in line.split()[2]:
                            dat_ev["UE-keypress"].append([int(line.split()[0]),line.split()[3][0]])
                            
                else:
                    if ("ET_REM" in line) or ("ET_CNT" in line):
                        if "001" in line.split()[2]:
                            dat_ev["Baseline"].append([int(line.split()[0]),line.split()[2]])
                            start = True
            f.close()
            initTime = dat_ev["Baseline"][0][0]
            
            for m in list(dat_ev.keys()):
                dat_ev[m] = [[int(((p[0] - initTime)/1000) / (1 / fs * 1000)), p[1]] for p in dat_ev[m]]
                
            events["UE-keypress"] = dat_ev["UE-keypress"]
        
        else:
            events["UE-keypress"] = [[p[0], p[1][1]] for p in events["UE-keypress"]]
        
        #%% ------------------ reject if no answer -------------------
        
        events[".avi"].append([len(pupilData)+1000,"dummy"])
                    
        rejectNum=[]
        rejectCondition=[]
        for iTrial in np.arange(len(events[".avi"])-1):
            tmp=[]
            
            for i,iRes in enumerate(events["UE-keypress"]):
                if events[".avi"][iTrial][0]+fs < iRes[0] < events[".avi"][iTrial+1][0]+fs:
                    tmp.append(i)
            if len(tmp) > 1:
                rejectNum.append(tmp[0:-1])
            if len(tmp) == 0:
                rejectCondition.append(iTrial)
                
        rejectNum = list(itertools.chain.from_iterable(rejectNum))
        events["UE-keypress"] = [d for i,d in enumerate(events["UE-keypress"]) if not i in rejectNum]
        events[".avi"] = [d for i,d in enumerate(events[".avi"]) if not i in rejectCondition]
           
        for iRes in events["UE-keypress"]:
            if iRes[1] == "B":
                datHash["Task"].append(0)
            elif iRes[1] == "X":
                datHash["Task"].append(0)
            elif iRes[1] == "M":
                datHash["Task"].append(1)
            else:
                datHash["Task"].append(1)
         
        #%% ------------------ data extraction ------------------       
        events[".avi"].pop(-1)     
        timeLen = int(cfg["WID_ANALYSIS"] * fs)
        rejectNum = []
        for ind in events[".avi"]:
            datHash["PDR"].append(pupilData[ind[0]:(ind[0]+timeLen)].tolist())
            datHash["gazeX"].append(gazeX[ind[0]:(ind[0]+timeLen)].tolist())
            datHash["gazeY"].append(gazeY[ind[0]:(ind[0]+timeLen)].tolist())
       
            datHash["zeroArray"].append(zeroArray[ind[0]:(ind[0]+timeLen)].tolist())
            datHash["baselinePD"].append(pupilData[(ind[0]-cfg["WID_BP_ANALYSIS"]*fs):ind[0]].tolist())
       
        
        for ind in events["UE-keypress"]:
            tmp = pupilData[ind[0]-int(1 * fs):(ind[0]+timeLen)]
            datHash["PDR_res"].append(tmp.tolist())
        
        #%% ------------------ condition frame ------------------
        condition = [str(e[1]) for e in events[".avi"]]
        tmp_c = np.zeros(len(events[".avi"]))
        
        for i in np.arange(NumOfconditions):
            t = inclusive_index(condition, name_condition[i])
            tmp_c[t] = i+1
            tmp_c[t] = int(name_condition[i].split("Lag")[1])
            
        datHash["SOA"] = np.r_[datHash["SOA"],tmp_c]
        
        #%% ------------------ RT -------------------------------    
        for iStartVideo,iKeyRes in zip(events[".avi"],events["UE-keypress"]):
           datHash["RT"].append((iKeyRes[0]-iStartVideo[0])*(1/fs)*1000 - 2500)
        
        #%% ------------------ sub number ----------------------- 
        datHash["sub"] = np.r_[datHash["sub"],np.ones(len(tmp_c),dtype=int)*int(folderList[iSub][1:3])]
        datHash["allAvePDR"].append(np.nanmean(pupilData))
        

    #%% ------------------ save data ----------------------------
    rejectNum = [i for i,p in enumerate(datHash["PDR"]) if len(p) < timeLen]
    
    e1 = datHash["allAvePDR"].copy()
    e2 = datHash["rejectFlag"].copy()
    for mm in list(datHash.keys()):
        datHash[mm] = [d for i,d in enumerate(datHash[mm]) if not i in rejectNum]
    
    #------------ reject gaze position ------------
    cfg["SAMPLING_RATE"] = cfg["RESAMPLING_RATE"]
    
    x = np.linspace(cfg["TIME_START"],cfg["TIME_END"],timeLen)
   
    t = 2.55
    gazeX = np.array(datHash["gazeX"])[:,getNearestValue(x,-0.5):getNearestValue(x,t)].tolist()
    gazeY = np.array(datHash["gazeY"])[:,getNearestValue(x,-0.5):getNearestValue(x,t)].tolist()
    
    fixations_list,angle_count, rejectGaze, fix_all = fixation_detection(datHash["sub"], gazeX, gazeY, cfg,[-0.5,t])
    
    datHash["theta"] = fixations_list["theta"].values.tolist()
    datHash["amp"] = fixations_list["amp"].values.tolist()    
    datHash["angle_count"] = angle_count
     
    center = np.array(cfg['SCREEN_RES'])/2
    
    datHash["gazeX"] = list(fixations_list["x"].values-center[0])
    datHash["gazeY"] = list(fixations_list["y"].values-center[1])
    
    fix_all["x"] = fix_all["x"] - center[0]
    fix_all["y"] = fix_all["y"] - center[1]
    
    cfg["rejectGaze"] = rejectGaze.tolist()
    
    if not os.path.exists("./data/"+date):
        os.mkdir("./data/"+date)
            
    with open("./data/"+date+"/dataOriginal"+sideRoop+unitName+".json","w") as f:
        json.dump(datHash,f)

    fix_all.reset_index(inplace=True)

    path = os.path.join("./data/"+date+"/fix_all"+sideRoop+unitName+".json")
    fix_all.to_json(path)
    
    with open(os.path.join("./data/"+date+"/cfg_parseData"+sideRoop+unitName+".json"),"w") as f:
        json.dump(cfg,f)
        
