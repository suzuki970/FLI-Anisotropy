#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:22:19 2022

@author: yutasuzuki

Input:    
cfg                    - dict of parameters for analysis
- "SAMPLING_RATE"      - int of sampling rate of gaze dat
- "DOT_PITCH"          - dot pitch of the used monitor
- "VISUAL_DISTANCE"    - visual distance from screen to eye
- "acceptMSRange"      - relative MS range threshold
- "SCREEN_RES"         - screen resolution
- "SCREEN_MM"          - screen size in mm
    
- "TIME_START" 
- "TIME_END"

cfg = {"SAMPLING_RATE":60,
       "TIME_START":-2,
       "TIME_END":3.5,
       "MIN_DURATION":150,
       "SCREEN_RES":[1680, 1050],
       "SCREEN_MM":[473.76, 296.1]
       }

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from tqdm import tqdm
from pixel_size import pixel_size
import pandas
from pre_processing import re_sampling

def distance2p(x1,y1,x2,y2):
    dx = x2 - x1
    dy = y2 - y1
    return np.sqrt((dx**2)+(dy**2))

def fixation_detection(sub, gazeX, gazeY, cfg, timeWin):
    
    center = np.array(cfg["SCREEN_RES"])/2

    s = timeWin[0]
    e = timeWin[1]
    
    fixList = pd.DataFrame(columns=["sub","trial","id", "x", "y", "dur"])
    fixList_all = pd.DataFrame(columns=["sub","trial","id", "x", "y", "dur"])
    
    angle_count=[]
    scthr = pixel_size(cfg["DOT_PITCH"],1,cfg["VISUAL_DISTANCE"])
        
    if cfg["SAMPLING_RATE"] > 100:
        gazeX = re_sampling(gazeX,int((e-s)*100)).tolist()
        gazeY = re_sampling(gazeY,int((e-s)*100)).tolist()
        fs = 100
    else:
        fs = cfg["SAMPLING_RATE"]
        
    n  =  len(gazeX[0])
    
    for iTrial, (iSub, gx, gy) in enumerate(tqdm(zip(sub,gazeX,gazeY))):
            
        fixations  =  np.zeros(n)
        
        #%% initialize pointers
        fixid = 0 # fixation id
        mx = 0    # mean coordinate x
        my = 0    # mean coordinate y
        d = 0     # dinstance between dat point and mean point
        fixpointer = 0 # fixations pointer
        for i in np.arange(1,n):
            # if (gx[i] > 0) & (gy[i] > 0):
            tmp_x = np.array(gx[fixpointer:i])
            tmp_y = np.array(gy[fixpointer:i])
            
            # mx = np.mean(tmp_x[tmp_x>0])
            # my = np.mean(tmp_y[tmp_x>0])
            mx = np.mean(tmp_x)
            my = np.mean(tmp_y)
            d = distance2p(mx,my,gx[i],gy[i])
            # print(d)
            if d > scthr: # gaze shifts more than > 1deg are regarded as saccades
                fixid = fixid + 1
                fixpointer = i
                    
            fixations[i] = fixid
                
            # number_fixations = fixations[n-1] # number of fixation after clustering (t1)
        
        #%% reject if duration is less than cfg["MIN_DURATION"]
        id_count=1
        tmp_fixList = pd.DataFrame(columns=["sub","trial","id", "x", "y", "dur"])
        
        for iFix in np.arange(max(fixations)+1):
            ind = np.argwhere(fixations == iFix).reshape(-1)
            
            tmp_x = np.array(gx)[ind]
            tmp_y = np.array(gy)[ind]
            
            if (len(ind) > cfg["MIN_DURATION"]*fs) & (tmp_x.mean() != 0):
                f = pd.DataFrame(
                    # np.array([iTrial, id_count, tmp_x[tmp_x>0].mean(), tmp_y[tmp_y>0].mean(), len(ind)]).reshape(1, 5),
                    np.array([iSub, iTrial, id_count, tmp_x.mean(), tmp_y.mean(), len(ind)]).reshape(1, 6),
                    columns=["sub","trial","id", "x", "y", "dur"])
                tmp_fixList =  pd.concat([tmp_fixList,f])
                # fixList = pd.concat([fixList,f])
                id_count+=1
        
        #%% amplitude and angle of saccades
        if len(tmp_fixList) > 0:
            xx = tmp_fixList["x"].values[0]-center[0]
            yy = tmp_fixList["y"].values[0]-center[1]
            amp = [np.sqrt(xx**2+yy**2)]
            theta = [np.arctan2(yy, xx) * (180 / np.pi)]
         
            for iFix in np.arange(len(tmp_fixList)-1):
                xx = tmp_fixList["x"].values[iFix+1]-tmp_fixList["x"].values[iFix]
                yy = tmp_fixList["y"].values[iFix+1]-tmp_fixList["y"].values[iFix]
                amp.append(np.sqrt(xx**2+yy**2))
                theta.append(np.arctan2(yy, xx) * (180 / np.pi))
             
            tmp_fixList["amp"]=amp
            tmp_fixList["theta"]=theta
             
            breaks = np.concatenate([np.array([0,15]),np.arange(45,350,30),[360]])
            tmp = pandas.cut(np.round(np.mod(theta,360)), bins=breaks)
           
            fixList_all =  pd.concat([fixList_all,tmp_fixList])
        
            #%% average with in trial
            tmp_fixList_ave = tmp_fixList.groupby(["trial"]).agg(np.nanmean)
            tmp = np.array(tmp.value_counts())
            tmp[0] = tmp[0]+tmp[-1]
            
            angle_count.append(tmp[:-1].tolist())
            
            if len(tmp_fixList_ave) > 0:
                tmp_fixList_ave["x"] = sum(tmp_fixList["x"] * tmp_fixList["dur"] ) / sum(tmp_fixList["dur"].values)
                tmp_fixList_ave["y"] = sum(tmp_fixList["y"] * tmp_fixList["dur"] ) / sum(tmp_fixList["dur"].values)
            
                fixList =  pd.concat([fixList,tmp_fixList_ave])
            else:
                f = pd.DataFrame(
                    np.zeros(tmp_fixList.shape[1]).reshape(1, tmp_fixList.shape[1]),
                    columns = list(tmp_fixList.keys()))
                fixList =  pd.concat([fixList,f])
        
        else:
            
            f = pd.DataFrame(
                np.zeros(tmp_fixList.shape[1]).reshape(1, tmp_fixList.shape[1]),
                columns = list(tmp_fixList.keys()))
            
            angle_count.append([])
            
            fixList_all =  pd.concat([fixList_all,tmp_fixList])
            fixList =  pd.concat([fixList,f])

        # fixList_ave = fixList.groupby(["trial"]).agg(np.nanmean)
    
    #%%
    thFix = pixel_size(cfg["DOT_PITCH"],30,cfg["VISUAL_DISTANCE"])  # changes in velocity of x > 30°/sec
    thAccFix = thFix*400   # changes in acceration of x > 1200°/sec
    gazeX = fixList["x"].values
    gazeY = fixList["y"].values
   
    rangeWin = pixel_size(cfg["DOT_PITCH"],cfg["acceptMSRange"],cfg["VISUAL_DISTANCE"])
    
    gazeX = gazeX-center[0]
    gazeY = gazeY-center[1]
    
    a = rangeWin**2
    b = rangeWin**2
    
    tmp_x = gazeX**2
    tmp_y = gazeY**2
    
    P = (tmp_x/a)+(tmp_y/b)-1
    
    rejectGaze = np.argwhere(P > 0).reshape(-1)
        
    plt.figure()
    ax = plt.axes()
    e = patches.Ellipse(xy=(0,0), width=rangeWin*2, height=rangeWin*2, fill=False, ec="r")
    ax.add_patch(e)
    plt.plot(gazeX,gazeY,".")
    plt.plot(gazeX[rejectGaze],gazeY[rejectGaze],"r.")
    plt.xlim([-400,400])
    plt.ylim([-400,400])
    ax.set_aspect("equal", adjustable="box")
        
    return fixList, angle_count, rejectGaze, fixList_all