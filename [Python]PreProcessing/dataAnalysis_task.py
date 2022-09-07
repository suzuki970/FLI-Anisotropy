#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 18:44:47 2021

@author: yutasuzuki
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import datetime
import glob
import seaborn as sns
import warnings
warnings.simplefilter('ignore')

sns.set()
sns.set_style('whitegrid')
sns.set_palette("Set2")

dt_now = datetime.datetime.now()
date = dt_now.strftime("%Y%m%d")

folderName = glob.glob(os.path.join("./data/*"))
folderName.sort()

folderName = folderName[-1]

#%% ------------------ initial settings --------------------------------------
cfg={
    "SAMPLING_RATE":60,
    "numOfTertile":2,
    "correct":True,
    "WID_BASELINE":np.array([-0.5,0]),
    "WID_FILTER":np.array([]),
    "METHOD":1, #subtraction,
    "FLAG_LOWPASS":False,
    "visualization":False
}

f = open(folderName + "/cfg_parseDataLVF_norm.json")
cfg.update(json.load(f))
f.close()

if cfg["normFlag"]:
    normName = "_norm"
else:
    normName = "_mm"    
    
#%%
for ivField in ["LVF","RVF"]:   
    
    #%% ------------------ data loading  -----------------------------------------   
    fileName = glob.glob(folderName+"/dataOriginal"+ivField+normName+"*")
    print(fileName[-1]+" loading...")
   
    fileName = fileName[-1]
    
    f = open(os.path.join(str(fileName)))
    dat = json.load(f)
    f.close()  
    
    reject = {}
  
    #%% ------- reject participant if the rejected trial were more than 30 -------
    numOfTrial = []
    numOfTrialAll = []
    
    for iSub in np.unique(dat["sub"]):
        cond=[]
        for iCondition in np.unique(dat["SOA"]):
            ind = np.argwhere((np.array(dat["sub"]) == iSub) & (np.array(dat["SOA"]) == iCondition)).reshape(-1)
            cond.append(len(ind))
        numOfTrial.append(cond)
        numOfTrialAll.append(sum(cond))
   
    #%% ------------------ make data frame ---------------------------------------
    df_tmp = pd.DataFrame()
    for mmName in ["sub","SOA","RT","Task"]:
        df_tmp[mmName] = dat[mmName]
    
    df_tmp["vField"] = ivField
    
    if ivField == "LVF":
        df = df_tmp.copy()
    else:
        df = pd.concat([df, df_tmp])
    
#%% ------------------ plot data(Accuracy) -------------------
if cfg["correct"]:
    
    #### They misunderstood the button response and perception
    for iSub in [11,31,42,44,45,49,52,53,66,68]:
        df["Task"][df["sub"]==iSub] = 1 - df["Task"][df["sub"]==iSub]
        # df["task_sorted"][df["sub"]==iSub] = 1 - df["task_sorted"][df["sub"]==iSub]
    
    #### They misunderstood only LVF
    for iSub in [19,20,70,75]:
        df["Task"][(df["sub"]==iSub) & (df["vField"]=="LVF")] = 1 - df["Task"][(df["sub"]==iSub) & (df["vField"]=="LVF")]
    
    #### They misunderstood only RVF
    df["Task"][(df["sub"]==54) & (df["vField"]=="RVF")] = 1 - df["Task"][(df["sub"]==54) & (df["vField"]=="RVF")]
     
df.reset_index(inplace=True)

#%% ------------------ save data -------------------
           
path = os.path.join(folderName+"/data_task"+normName+".json")
df.to_json(path)

#%% ------------------ [plot] tertile in each VFs-------------------

g0 = ["b^","r^"]
jitter = [-1,1]
x = np.array([-83.3,-66.7,-50.0,-33.3,-16.7,0,16.7,33.3])

df["Task"][df["vField"]=="RVF"] = 1 - df["Task"][df["vField"]=="RVF"]
 
#%% ------------------ [plot] in each VFs -------------------  
df_tmp = df.groupby(["sub","SOA","vField"], as_index=False).agg(np.nanmean)

plt.figure()
sns.pointplot(x="SOA", y='Task', hue = "vField", data = df_tmp, dodge=True)
plt.legend()
