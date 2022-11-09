
import numpy as np
import matplotlib.pyplot as plt
from pre_processing_cls import pre_processing,getNearestValue,rejectDat,rejectedByOutlier,MPCL
from rejectBlink_PCA import rejectBlink_PCA
import json
import os
import pandas as pd
from scipy import stats
import datetime
import glob
import seaborn as sns
import warnings
from statsmodels.stats.anova import AnovaRM

warnings.simplefilter("ignore")

sns.set()
sns.set_style("whitegrid")
sns.set_palette("Set2")

dt_now = datetime.datetime.now()
date = dt_now.strftime("%Y%m%d")

folderName = glob.glob(os.path.join("./data/*"))
folderName.sort()

folderName = folderName[-1]
  
df = pd.DataFrame()

#%% ------------------ initial settings --------------------------------------
cfg={
    "SAMPLING_RATE":60,   
    "windowL":[2],
    "numOfTertile":2,
    "correct":True,
    "WID_BASELINE":[[-0.5,0]],
    "WID_FILTER":np.array([]),
    "METHOD":1, #subtraction,
    "FLAG_LOWPASS":False,
    "visualization":False
}
    
f = open(folderName + "/cfg_parseDataLVF_norm.json")
cfg.update(json.load(f))
f.close()

if not cfg["mmFlag"] and not cfg["normFlag"]:
    unitName = "_au"
    cfg["THRES_DIFF"] = 0.15
elif cfg["mmFlag"]:
    unitName = "_mm"
    cfg["THRES_DIFF"] = 0.15
else:
    unitName = "_norm" 
    cfg["THRES_DIFF"] = 0.3

#%%
for ivField in ["LVF","RVF"]:   
     
    #%% ------------------ data loading  -----------------------------------------   
    fileName = glob.glob(folderName+"/dataOriginal"+ivField+unitName+"*")
    print(folderName+" loading...")
   
    fileName = fileName[-1]
       
    f = open(os.path.join(str(fileName)))
    dat = json.load(f)
    f.close()

    #%% ------------------ cfg loading  -----------------------------------------   

    f = open(folderName + "/cfg_parseData"+ivField+unitName+".json")
    cfg.update(json.load(f))
    f.close()

    cfg_resLocs = cfg.copy()
    cfg_resLocs["TIME_START"] = -1
    cfg_resLocs["TIME_END"] = cfg["WID_ANALYSIS"]
   
    reject = {}
    pp = pre_processing(cfg)
    pp_resLocs = pre_processing(cfg_resLocs)
    
    dat["Baseline"] = dat["PDR"]

    gazeX = np.array(dat["gazeX"])
    gazeY = np.array(dat["gazeY"])
    
    x = np.linspace(cfg["TIME_START"],cfg["TIME_END"],np.array(dat["PDR"]).shape[1])
    
    blink = np.array(dat["zeroArray"])
    blink = blink.mean(axis=0)
    
    #%% ------------------ data-loss detection (>80%)  -----------------------------------------   

    ind = np.argwhere(blink > 0.8)
    blink_ind = x[ind]

    cfg["WID_ANALYSIS"] = np.float(x[ind[0]-1])
    
    df_allPDR_tmp = pd.DataFrame()
    for mmName in ["allAvePDR"]:
        df_allPDR_tmp[mmName] = dat[mmName]
        
    df_allPDR_tmp["vField"] = ivField
    
    #%% ------------------ rejection by gaze ------------------------------------
    
    dat = rejectDat(dat,cfg["rejectGaze"])
    
    #%% ------------------ artifact rejection ------------------------------------
    y,reject["thre"]  = pp.pre_processing(np.array(dat["PDR"]))
    
    rejectNum = [i for i,p in enumerate(dat["PDR_res"]) if len(p) < 393]
    dat, y = rejectDat(dat,rejectNum,y)
  
    y2, reject["thre_resLock"] = pp_resLocs.pre_processing(np.array(dat["PDR_res"]))
    dat["PDR_res"] = y2.tolist()
    
    reject["thre"] = rejectedByOutlier(dat, np.array(dat["PDR"])[:,getNearestValue(x,cfg["WID_BASELINE"][0][0]):getNearestValue(x,cfg["WID_ANALYSIS"])].mean(axis=1))
                                       
    dat, y = rejectDat(dat,reject["thre"],y)
  
    #%% ------------------ PCA ---------------------------------------------------
    pca,reject["PCA"] = rejectBlink_PCA(y)
    
    dat, y = rejectDat(dat,reject["PCA"].tolist(),y)
   
    #%% ------------------ baseline ----------------------------------------------
   
    y_bp = np.array(dat["PDR"])[:,getNearestValue(x,cfg["WID_BASELINE"][0]):getNearestValue(x,cfg["WID_BASELINE"][1])].mean(axis=1)
    
    
    
    dat["Baseline"] = y_bp.tolist()
    
    #%% ------------------ reject participant if the rejected trial were more than 40% -------
    numOfTrial = []
    tmp_reject=[]
    NUM_TRIAL = 80
    subName = np.unique(dat["sub"])
    
    for isub in np.arange(len(subName)):
        cond=[]
        for iCondition in np.unique(dat["SOA"]):
            tmp = [i for i,(s,c) in enumerate(zip(dat["sub"],dat["SOA"])) if s == subName[isub] and c == iCondition]
            cond.append(len(tmp))
        numOfTrial.append(cond)
        
        if sum(numOfTrial[isub]) < NUM_TRIAL * 0.6:
            tmp_reject.append(subName[isub])
        
    reject["sub"] = [i for i,d in enumerate(dat["sub"]) if d in tmp_reject]
    print("Subject = " + str(tmp_reject) + " rejected by N")
   
    dat, y,  y_bp = rejectDat(dat,reject["sub"],y,y_bp)
    
    #%% ------------------ save data ---------------------------------------------
    
    if cfg["correct"]:
        dat["Task"] = np.array(dat["Task"])
        
        #### They misunderstood the button response and perception
        for iSub in [11,31,42,44,45,49,52,53,66,68]:
            ind = np.argwhere(np.array(dat["sub"])==iSub).reshape(-1)
            dat["Task"][ind] = 1 - dat["Task"][ind]
        
        #### They misunderstood only LVF
        if ivField == "LVF":
            for iSub in [19,20,70,75]:
                ind = np.argwhere(np.array(dat["sub"])==iSub).reshape(-1)
                dat["Task"][ind] = 1-dat["Task"][ind]
         
        #### They misunderstood only RVF
        if ivField == "RVF":
            ind = np.argwhere(np.array(dat["sub"])==54).reshape(-1)
            dat["Task"][ind] = 1-dat["Task"][ind]
            
        dat["Task"] = dat["Task"].tolist()
    
    else:
        rejectSub = [11,31,42,44,45,49,52,53,54,66,68,70,75]
        
        rejectSubInd = [i for i,d in enumerate(dat["sub"]) if d in rejectSub]
       
        y = np.delete(y,rejectSubInd,axis=0)  
        y_bp = np.delete(y_bp,rejectSubInd,axis=0)
        for mm in list(dat.keys()):
            dat[mm] = [d for i,d in enumerate(dat[mm]) if not i in rejectSubInd]
   
        dat["Task"] = [int(i) for i in dat["Task"]]
    
    dat["PDR"] = y.tolist()
    
    if not os.path.exists("./data/"+date):
            os.mkdir("./data/"+date)
            
    with open("./data/"+date+"/data"+ivField+unitName+".json","w") as f:
        json.dump(dat,f)
 
   
    #%% ------------------ make data frame ---------------------------------------
    x = np.linspace(cfg["TIME_START"],cfg["TIME_END"] ,np.array(dat["PDR"]).shape[1])

    df_tmp = pd.DataFrame()
    for mmName in ["sub","SOA","RT","Task","Baseline","gazeX","gazeY" ,"theta","amp" ]:
        df_tmp[mmName] = dat[mmName]

    df_tmp["EventPD"] = np.array(dat["PDR"])[:,getNearestValue(x,0):getNearestValue(x,cfg["WID_ANALYSIS"])].mean(axis=1)
    
    df_tmp["vField"] = ivField
    
    if ivField == "LVF":
        df = df_tmp.copy()
        df_allPDR = df_allPDR_tmp.copy()
        y_LVF = y
    else:
        df = pd.concat([df, df_tmp])
        
        df_allPDR = pd.concat([df_allPDR, df_allPDR_tmp])
    
df["Task"][df["vField"]=="RVF"] = 1 - df["Task"][df["vField"]=="RVF"]


#%% ------------------ save data -------------------

df.reset_index(inplace=True)

reject["rejectedOneVF"] = list(set(df[df["vField"]=="LVF"]["sub"]) ^ set(df[df["vField"]=="RVF"]["sub"]))

y_rl = np.r_[y_LVF,y]

if len(reject["rejectedOneVF"]) > 0:
    for rej in reject["rejectedOneVF"]:
        ind = np.argwhere(df["sub"].values == rej).reshape(-1)
        y_rl = np.delete(y_rl,ind,axis=0)
        df = df[df["sub"]!=rej]
    

tmp_mpcl = MPCL(df["sub"].values, y_rl, cfg, "PD", 0, cfg["WID_ANALYSIS"] )
cfg["maxVal"] = list(tmp_mpcl["xval"].values)

tmp_mpcl = MPCL(df["sub"].values, y_rl, cfg, "PC", cfg["maxVal"], cfg["WID_ANALYSIS"] )
cfg["minVal"] = list(tmp_mpcl["xval"].values)


path = os.path.join("./data/"+date+"/data_df"+unitName+".json")
    
df.to_json(path)

with open(os.path.join("./data/"+date+"/cfg"+unitName+".json"),"w") as f:
    json.dump(cfg,f)


#%% ------------------ plot data(Accuracy) -------------------

df_tmp = df.groupby(["sub","SOA","vField"], as_index=False).agg(np.nanmean)

# ------------------ average ------------------
plt.figure(figsize=(7,7))
plt.title("Tertile", fontsize=20)

sns.pointplot(x="SOA", y="Task", hue = "vField", data=df_tmp, dodge=True)
plt.xlabel("Lag[ms]",fontsize=14)
plt.ylabel("Accura cy",fontsize=14)
plt.ylim(0, 1)

#%% ------------------ show data (Baseline pupil size) ------------------

plt.figure()
sns.pointplot(x="SOA", y="Baseline", hue = "vField", data=df_tmp, dodge=True)
plt.xlabel("Lag[ms]",fontsize=14)
plt.ylabel("pupil diameter [mm]",fontsize=14)
plt.legend()


#   ------------------ ave across conditions ------------------
df_ave_ind = df.groupby(["sub","vField"], as_index=False).agg(np.nanmean)

test_x = df_ave_ind[df_ave_ind["vField"]=="LVF"]["Baseline"].values
test_y = df_ave_ind[df_ave_ind["vField"]=="RVF"]["Baseline"].values

res = stats.ttest_ind(test_x, test_y)[1]

plt.figure(figsize=(7,10))
sns.pointplot(x="vField", y="Baseline", hue = "vField", data=df_ave_ind, dodge=True)
plt.tick_params(labelsize=18)
plt.title("p = " + str(np.round(res,4)))
plt.ylabel("pupil diameter [mm]",fontsize=14)

#%% ------------------ show data (each VF)-------------------
   
df_tmp = df.groupby(["sub","Task","SOA","vField"], as_index=False).agg(np.nanmean)

plt.figure()
sns.pointplot(x="SOA", y="EventPD", hue = "vField", data=df_tmp, dodge=True)
plt.xlabel("Lag[ms]",fontsize=14)
plt.ylabel("pupil diameter [mm]",fontsize=14)
plt.legend()

# ------------------ phasic ------------------
plt.figure()
grid = sns.FacetGrid(df_tmp, col="vField", hue="Task", size=5)
grid.map(sns.pointplot, "SOA", "EventPD", dodge=True)


#%% ------------------ velocity -------------------------------

x = np.linspace(cfg["TIME_START"],cfg["TIME_END"],y_rl.shape[1])

##### make data frame
df_timeCourse = pd.DataFrame()
df_timeCourse["y"] = y_rl.reshape(-1)
df_timeCourse["x"] = np.tile(x, y_rl.shape[0])
df_timeCourse["SOA"] = np.tile( df["SOA"].values.reshape(df["SOA"].shape[0],1), y_rl.shape[1]).reshape(-1)
df_timeCourse["vField"] = np.tile( df["vField"].values.reshape(df["vField"].shape[0],1), y_rl.shape[1]).reshape(-1)
df_timeCourse["sub"] = np.tile( df["sub"].values.reshape(df["sub"].shape[0],1), y_rl.shape[1]).reshape(-1)

df_timeCourse = df_timeCourse.groupby(["sub","vField","SOA","x"], as_index=False).agg(np.nanmean)

df_vel = df_timeCourse.groupby(["sub","vField","SOA"], as_index=False).agg(np.nanmean)

tmp_vel = []
for i,iSub in enumerate(np.unique(df_timeCourse["sub"])):
    for ivField in np.unique(df_timeCourse["vField"]):
        for iSOA in np.unique(df_timeCourse["SOA"]):
            
            tmp = df_timeCourse[(df_timeCourse["sub"]==iSub) & 
                                (df_timeCourse["SOA"]==iSOA) & 
                                (df_timeCourse["vField"] == ivField)]
            
            tmp_vel.append(np.diff(tmp[(tmp["x"] >= cfg["maxVal"][i]) & 
                                       (tmp["x"] <= cfg["minVal"][i])]["y"].values).mean()*cfg["SAMPLING_RATE"])
  
df_vel["vel"] = tmp_vel

plt.figure()
sns.pointplot(data=df_vel,x="SOA", y="vel", hue = "vField", ci="sd")
print(AnovaRM(data=df_vel, depvar="vel", subject="sub", within=["vField","SOA"]).fit())

df_vel.reset_index(inplace=True)
df_vel.to_json(os.path.join("./data/"+date+"/velocity"+unitName+".json"))

tmp = df_timeCourse.groupby(["vField","SOA","x"], as_index=False).agg(np.nanmean)

# -------------------- average time course --------------------
plt.figure()
plt.title("average pupil time course")
sns.lineplot(data=df_timeCourse, x="x", y="y", hue = "sub", ci="sd")

tmp_plot = df_timeCourse.groupby(["sub","x"], as_index=False).agg(np.nanmean)

sns.lineplot(data=tmp_plot,x="x", y="y", hue = "sub", ci=False)
sns.lineplot(data=tmp_plot,x="x", y="y", hue = "sub", ci=False)

df_minVal = pd.DataFrame()
df_minVal["sub"] = np.unique(df["sub"])
df_minVal["maxVal"] = cfg["maxVal"] 

