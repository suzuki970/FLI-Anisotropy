
import numpy as np
import numpy.matlib
from band_pass_filter import butter_bandpass_filter,lowpass_filter
import pandas
from scipy import interpolate,fftpack
from scipy.fftpack import fft2, ifft2
from scipy import signal,fftpack
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns

from scipy.stats import norm
import math
Z = norm.ppf

sns.set()
sns.set_style('whitegrid')
sns.set_palette("Set2")

class pre_processing:    
    
    def __init__(self, cfg):
        self.cfg = cfg
      
    #%%
    def pre_processing(self, dat, rejectFlg=False):
        """
        Input:    
            dat  - 
            cfg  -
        Output:
        Example:
        """
      
        #%% if data is list
        if isinstance(dat, list):
            rejectNum=[]
            out_y = []
            for i,p in enumerate(dat):
                
                timeWin = len(p)
                
                ## Smoothing
                s = pandas.Series(p)
                y = s.rolling(window=self.cfg['windowL']).mean()
                y = np.array(y)
                timeWin = len(y)
                
                x = np.linspace(self.cfg['TIME_START'][i],self.cfg['TIME_END'][i],timeWin)
               
                # filtering
                # if len(filt) > 0:
                #     ave = np.nanmean(y)
                #     y = y - ave
                #     y = butter_bandpass_filter(y, filt[0], filt[1], fs, order=4)
                #     y = y + ave
            
                baselineData = np.array([getNearestValue(x,self.cfg['WID_BASELINE'][0][0]),getNearestValue(x,self.cfg['WID_BASELINE'][0][1]),getNearestValue(x,self.cfg['WID_ANALYSIS'][i])])
                baselinePLR = y[np.arange(baselineData[0],baselineData[1])]
                baselinePLR_std = np.std(baselinePLR)
                # baselinePLR_std = np.tile(baselinePLR_std, (1,timeWin)).reshape(1,timeWin).T
                baselinePLR = np.mean(baselinePLR)
                # baselinePLR = np.tile(baselinePLR, (1,timeWin)).reshape(timeWin,dat.shape[0]).T   
               
                if self.cfg['METHOD'] == 1:
                    y = y - baselinePLR
                elif self.cfg['METHOD'] == 2:
                    y = (y - baselinePLR) / baselinePLR_std
                else:
                    y = y 
                    
                fx = np.diff(y)
                ind = np.argwhere(abs(fx[np.arange(baselineData[0],baselineData[2])]) > self.cfg['THRES_DIFF'])
                # ind = np.argwhere(abs(fx) > thres)
            
                if len(ind) > 0:
                    rejectNum.append(i)
                    # continue
                
                ## reject trials when number of 0 > 50#
                if sum(np.argwhere(y[np.arange(baselineData[0],baselineData[2])] == 0)) > y.shape[0] / 2:
                    rejectNum.append(i)
                    # continue
                    
                ## reject trials when the NAN includes
                if len(np.argwhere(np.isnan(y[self.cfg['windowL']:]) == True)) > 0:
                    rejectNum.append(i)
                    
                out_y.append(y)
                
            rejectNum = np.unique(rejectNum)
            set(rejectNum)
            y = out_y
            
       
            
        #%% if data is array
        else:
            ##### Smoothing
            if len(self.cfg['windowL']) > 0:
                y = moving_avg(dat.copy(),int(np.array(self.cfg['windowL'])))
            else:
                y = dat.copy()
                
            ##### baseline(-200ms - 0ms)
            x = np.linspace(self.cfg['TIME_START'],self.cfg['TIME_END'],y.shape[1])
          
            ###### filtering
            if len(self.cfg['WID_FILTER']) > 0:
                ave = np.mean(y,axis=1)
                y = y - np.tile(ave, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T
                y = butter_bandpass_filter(y, self.cfg['WID_FILTER'][0], self.cfg['WID_FILTER'][1], self.cfg['SAMPLING_RATE'], order=4)
                y = y + np.tile(ave, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T
        
            ##### baseline correction
            if len(self.cfg['WID_BASELINE']) > 2:
                tmp_baseline = np.zeros((y.shape[0],y.shape[1]))
                for iTrial in np.arange(len(self.cfg['WID_BASELINE'])):
                    baselineData = np.array([getNearestValue(x,self.cfg['WID_BASELINE'][iTrial][0]),getNearestValue(x,self.cfg['WID_BASELINE'][iTrial][1]),getNearestValue(x,self.cfg['WID_ANALYSIS'])])
                    baselinePLR = y[iTrial,np.arange(baselineData[0],baselineData[1])]
                    # baselinePLR_std = np.std(baselinePLR,axis=1)
                    # baselinePLR_std = np.tile(baselinePLR_std, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T
                    baselinePLR = np.mean(baselinePLR)
                    # baselinePLR = np.tile(baselinePLR, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T   
                    tmp_baseline[iTrial,:] = np.tile(baselinePLR, (1,y.shape[1]))
                baselinePLR = tmp_baseline
            else:
                baselineData = np.array([getNearestValue(x,self.cfg['WID_BASELINE'][0][0]),getNearestValue(x,self.cfg['WID_BASELINE'][0][1]),getNearestValue(x,self.cfg['WID_ANALYSIS'])])
                baselinePLR = y[:,np.arange(baselineData[0],baselineData[1])]
                baselinePLR_std = np.std(baselinePLR,axis=1)
                baselinePLR_std = np.tile(baselinePLR_std, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T
                baselinePLR = np.mean(baselinePLR,axis=1)
                baselinePLR = np.tile(baselinePLR, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T   
           
            if self.cfg['METHOD'] == 1: # subtraction
                y = y - baselinePLR
            elif self.cfg['METHOD'] == 2: # population
                # y = (y - baselinePLR) / baselinePLR_std
                y = y / baselinePLR
            else: 
                ave = np.mean(y,axis=1)
                ave = np.tile(ave, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T
                std = np.std(y,axis=1)
                std = np.tile(std, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T           
                y = (y - ave) / std
                
                baselinePLR = y[:,np.arange(baselineData[0],baselineData[1])]
                baselinePLR = np.mean(baselinePLR,axis=1)
                baselinePLR = np.tile(baselinePLR, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T   
                y = y - baselinePLR
            
            ##### low-pass filter
            if self.cfg['FLAG_LOWPASS']:
                y = lowpass_filter(y, self.cfg['TIME_END']-self.cfg['TIME_START'])
           
            ##### reject artifact
            rejectNum = reject_trials(y,self.cfg['THRES_DIFF'],baselineData)
            
            
        return y,rejectNum
    
     
    #%% Hippus
    def getHippus(self, dat, sec = 3, overlap = 90, zeropad = 2 ** 11):
    
        ave = np.mean(dat,axis=1)
        dat = dat - np.tile(ave, (1,dat.shape[1])).reshape(dat.shape[1],dat.shape[0]).T
        
        amp = []
        for iBP in tqdm(dat.tolist()):
            
            time_array, N_ave = ov(iBP, self.cfg["SAMPLING_RATE"], overlap, self.cfg["SAMPLING_RATE"]*sec)
            time_array, acf = hanning(time_array, N_ave, self.cfg["SAMPLING_RATE"]*sec)
            time_array = zero_padding(np.array(time_array.copy()), zeropad)
            
            spectrum, fft_array, phase, freq = getfft(time_array, self.cfg["SAMPLING_RATE"])
            amp.append(fft_array.mean(axis=0).tolist())
    
        amp = np.array(amp)[:,getNearestValue(freq, 0):getNearestValue(freq, 3)].tolist()
        freq = freq[getNearestValue(freq, 0):getNearestValue(freq, 3)].tolist()
        
        # plt.plot(freq,np.array(amp).mean(axis=0))
        # plt.plot(freq,np.array(amp).T)
        indPeak = np.argmax(np.array(amp).mean(axis=0))

        return amp,freq,np.array(amp)[:,indPeak].tolist()

    #%% Blink
    def getBlink(self, blink, ev_t1, jitter=[], fs=1000):
        
        ##### blink detection
        # downSampleRate = 100        
        buffer = int(0.05*fs)

        windowLen = self.cfg['WID_ANALYSIS']*2
        rate = np.zeros((len(blink),windowLen*fs))
        rate_tmp = np.zeros((len(blink),windowLen*fs))

        for iTrial,dTrial in enumerate(blink):
            for iBlink in dTrial:
                if iBlink[1] < (windowLen/2)*fs:
                    rate[iTrial,np.arange(iBlink[0],iBlink[1])+(self.cfg["WID_ANALYSIS"]*fs)] = 1
                    
                    if iBlink[1]+buffer < (self.cfg["WID_ANALYSIS"]*fs):
                        rate_tmp[iTrial,np.arange(iBlink[0]-buffer,iBlink[1]+buffer)+(self.cfg["WID_ANALYSIS"]*fs)] = 1
                    else:
                        rate_tmp[iTrial,(iBlink[0]-buffer)+(self.cfg["WID_ANALYSIS"]*fs):] = 1
                   
        # rate_tmp = re_sampling(rate_tmp,int((windowLen)*fs))
        rate_tmp[rate_tmp==-1]=0
        rate_tmp[rate_tmp==2]=1
        
        if len(jitter) > 0:
            for iTrial,lag in enumerate(jitter):
                rate_tmp[iTrial,:] = np.r_[rate_tmp[iTrial,int(lag):],np.zeros(int(lag))]

        ##### reject if MS overlap with blink
        coeff = 1-rate_tmp.copy()

        sTimeOfMS = []
        for iTrial,dTrial in enumerate(ev_t1):
            rejectMS = []
            empty = np.zeros(coeff.shape[1])
                
            for ims,dms in enumerate(dTrial):
                t = coeff[iTrial,dms["onset"]:dms["offset"]]
                if len(np.argwhere(t==0)):
                    rejectMS.append(ims)
                
                if dms["onset"]>0:
                    if dms["amplitude"] > 0:
                        empty[dms["onset"]] = 1
                    else:
                        empty[dms["onset"]] = -1
                        
            ev_t1[iTrial] = [d for i,d in enumerate(dTrial) if not i in rejectMS]

            sTimeOfMS.append(empty.tolist())

        sTimeOfMS = np.multiply(np.array(sTimeOfMS),coeff)
       
        ##### moving average
        tmp = moving_avg(sTimeOfMS,int(fs/10))
        sTimeOfMS = re_sampling(tmp, int((self.cfg['TIME_END']-self.cfg['TIME_START'])*self.cfg["RESAMPLING_RATE"])).tolist()
    
        dat_MS = {"amplitude":[],
                     "theta":[],
                     "onset":[],
                     "trial":[]
                     }
        
        for iTrial,dTrial_t1 in enumerate(ev_t1):
            rejectMS = []
            empty = np.zeros(coeff.shape[1])
            
            for ims,dms in enumerate(dTrial_t1):
                dat_MS["amplitude"].append(dms["amplitude"])
                dat_MS["theta"].append(dms["theta"])
                dat_MS["onset"].append(dms["onset"]/fs)
                dat_MS["trial"].append(iTrial)
      
        rate = moving_avg(rate.copy(),100)
        rate = re_sampling(rate,int((windowLen)*self.cfg["RESAMPLING_RATE"]))
        
        return rate, dat_MS, sTimeOfMS
    

#%% Add vertical lines for mean age on each plot
def vertical_line(x, **kwargs):
    plt.axvline(x.mean(), linestyle = '--', color = 'g')


#%% overlap
def ov(data, fs, overlap, frame = 2**12):
    """
    Input:    
    Output:
    Example:
    """
    N = len(data)  
    Ts = N / fs         #data
    Fc = frame / fs     #frame
    x_ol = frame * (1 - (overlap/100)) #overlap
    N_ave = int((Ts - (Fc * (overlap/100))) / (Fc * (1-(overlap/100))))

    array = []

    for i in range(N_ave):
        ps = int(x_ol * i) 
        # dat = zero_padding(data[ps:ps+frame:1],zeropad)
        # array.append(dat) 
        array.append(data[ps:ps+frame:1]) 
        
    return array, N_ave

#%%
def hanning(data, N_ave, frame = 2**12):
    """
    Input:    
    Output:
    Example:
    """
    #haning
    # han = signal.hann(frame) 
    han = signal.kaiser(frame,2)
    # han = signal.blackman(frame)
    
    acf = 1 / (sum(han) / frame)
    
    data_pad = []
    for i in range(N_ave):
        data[i] = data[i] * han
        data_pad.append(data[i])
        
    return data_pad, acf

#%% FFT
def fft_ave(data,fs, N_ave, acf, frame = 2**12):
    
    """
    Input:    
    Output:
    Example:
    """
    fft_array = []
    for i in range(N_ave):
        fft_array.append(acf*np.abs(fftpack.fft(data[i])/(frame/2)))
        
    fft_axis = np.linspace(0, fs, frame)
    fft_array = np.array(fft_array)
    fft_mean = np.sqrt(np.mean(fft_array ** 2, axis=0))
    
    return fft_array, fft_mean, fft_axis

#%%
def zero_padding(data, len_pad):
    """
    Input:    
    Output:
    Example:
    """
    if data.ndim == 1:
        pad = np.zeros(len_pad)
        data_pad = np.hstack((np.insert(data, 0, pad), pad))
        acf = (sum(np.abs(data)) / len(data)) / (sum(np.abs(data_pad)) / len(data_pad))
        return data_pad * acf
    else:    
        pad = np.zeros((data.shape[0],len_pad))
        data_pad = np.hstack([np.hstack([pad,data]),pad])
        acf = (np.abs(data[0,:]).sum() / data.shape[1]) / (np.abs(data_pad[0,:]).sum() / data_pad.shape[1])
        
        return data_pad

#%%
def split_list2(l, n):
    windowL = np.round(np.linspace(0, len(l), n+1))
    windowL = [int(windowL[i]) for i in np.arange(len(windowL))]
    for idx in np.arange(len(windowL)-1):
        yield np.arange(windowL[idx],windowL[idx+1]).tolist(),l[windowL[idx]:windowL[idx+1]]

#%%
def split_list(l, n):
    windowL = np.round(np.linspace(0, len(l), n+1))
    windowL = [int(windowL[i]) for i in np.arange(len(windowL))]
    for idx in np.arange(len(windowL)-1):
        yield l[windowL[idx]:windowL[idx+1]]
  
#%% perform fourier transform
def getfft(data, fs):
    # spectrum = fftpack.fft(data)
    # spectrum = []
    # for d in data:
    #     spectrum.append(fftpack.fft(d))
    spectrum = np.fft.fft(data, axis=1, norm=None)   
    spectrum = np.array(spectrum)
                     
    N = len(data[0])
    amp = abs(spectrum)
    amp = amp / (N / 2)
    phase = np.arctan2(spectrum.imag, spectrum.real)
    phase = np.degrees(phase)
    freq = np.linspace(0, fs, N)

    return spectrum[:,:round(N / 2)], amp[:,:round(N / 2)], phase[:,:round(N / 2)], freq[:round(N / 2)]

#%%
def rejectDat(dat,rejectNum,*args):
    for mm in list(dat.keys()):
        dat[mm] = [d for i,d in enumerate(dat[mm]) if not i in rejectNum]
    
    if len(args) > 0:
        tmp = []
        for i,arg in enumerate(args):
            tmp.append(np.delete(arg,rejectNum,axis=0))
    
        return dat, *tuple(tmp)

    else: 
        return dat
    
#%% zscore
def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

#%% re_sampling
def re_sampling(dat,num):
    """
    Input:    
        dat  - list of data
        num  - list or int of re-sampling rate
    Output:
    Example:
    """
    re_sampled = []
    for iTrial,d in enumerate(dat):
        t = np.array(d)
        numX = np.arange(len(t))
        yy = interpolate.PchipInterpolator(numX, t)
        if isinstance(num, list):
            t_resample = np.linspace(0, len(t), num[iTrial])
        else:
            t_resample = np.linspace(0, len(t), num)
            
        re_sampled.append(yy(t_resample))
    
    return np.array(re_sampled)

#%% getNearestValue
def getNearestValue(in_y, num):
    idx = np.abs(np.asarray(in_y) - num).argmin()
    return idx

#%% moving_avg
def moving_avg(x,windowL):
    
    """
    Input:    
        dat      - list of data
        windowL  - list or int of re-sampling rate
    Output:
    Example:
    """
    if isinstance(x, list):
        tmp_y = x.copy()
        
        for iTrials,d in enumerate(x):
            s = pandas.Series(np.array(d))
            x[iTrials] = np.array(s.rolling(window=windowL).mean())
        
        out_y = []
        for iTrials,d in enumerate(x):
            out_y.append(np.r_[np.array(tmp_y[iTrials])[np.arange(windowL)],
                         np.array(d)[np.arange(windowL,len(d))]].tolist())
            
    else:
    
        if x.ndim == 1:
            x = x.reshape(1,len(x))
            
        tmp_y = x.copy()
        
        for trials in np.arange(len(x)):
            s = pandas.Series(x[trials,])
            x[trials,] = s.rolling(window=windowL).mean()
        
        out_y = []
        for trials in np.arange(len(x)):
            out_y.append(np.r_[np.array(tmp_y[trials])[np.arange(windowL)],
                         x[trials,np.arange(windowL,x.shape[1])]])
        out_y = np.array(out_y)

    return out_y

#%%
def reject_trials(y,thres,baselineData):
  
    ## reject trials when the velocity of pupil change is larger than threshold
    rejectNum=[]
    fx = abs(np.diff(y, n=1))
    fx = fx[:,np.arange(baselineData[0],baselineData[2])]
    
    ind = np.argwhere(fx > thres)

    rejectNum = np.unique(ind[:,0])

    
    # for trials in np.arange(y.shape[0]):
    #     ind = np.argwhere(abs(fx[trials,np.arange(baselineData[0],baselineData[2])]) > thres)
    #     # ind = np.argwhere(abs(fx[trials,np.arange(50,baselineData[2])]) > thres)
        
    #     if len(ind) > 0:
    #         plt.plot(fx[trials,:])
    #         rejectNum.append(trials)
    #         continue
            
    #     if sum(np.isnan(y[trials,np.arange(baselineData[0],baselineData[2])])) > 0:
    #         rejectNum.append(trials)
    #         continue
    
        ## reject trials when number of 0 > 50#
        # if sum(np.argwhere(y[trials,np.arange(baselineData[0],baselineData[2])] == 0)) > y.shape[0] / 2:
        #     rejectNum.append(trials)
        #     continue
        
    ## reject trials when the NAN includes
    # tmp = np.argwhere(np.isnan(y) == True) 
    # for i in np.arange(tmp.shape[0]):
    #     rejectNum.append(tmp[i,0])
        
    rejectNum = np.unique(rejectNum)
    set(rejectNum)
    return rejectNum.tolist()

#%%
def rejectedByOutlier(dat, y_bp, factorName = "BaselinePD"):
    
    if y_bp.ndim != 1:
        y_bp = y_bp.mean(axis=1)
        # y_bp = np.max(np.abs(y_bp),axis=1)
       
    tmp = pd.DataFrame()
    tmp["sub"] =np.array(dat["sub"])
    tmp[factorName] = y_bp
     
    df_std = tmp.groupby(["sub"]).agg(np.std)
    df_ave = tmp.groupby(["sub"]).agg(np.median)
        
    df_std[factorName] = df_std[factorName]*3

    df_ave["lower"] = df_ave[factorName] - df_std[factorName] 
    df_ave["upper"] = df_ave[factorName] + df_std[factorName] 

    tmp["lower"] = 0
    tmp["upper"] = 0
    for iSub, lo, up in zip(np.unique(dat["sub"]),list(df_ave["lower"]), list(df_ave["upper"])):
        tmp["lower"][tmp["sub"]==iSub] = lo
        tmp["upper"][tmp["sub"]==iSub] = up
        
    reject = tmp[(tmp[factorName] < tmp["lower"]) | (tmp[factorName] > tmp["upper"])].index.values

    grid = sns.FacetGrid(tmp, col="sub", hue="sub", col_wrap=5, size=5)
    grid.map(plt.hist, factorName, bins=30)

    grid.map(vertical_line, 'lower') 
    grid.map(vertical_line, 'upper') 

    # dat, y, y2, y_bp = rejectDat(dat,reject["BP"].tolist(),y,y2,y_bp)
  
    return reject

#%%
def rejectedSubByOutlier(dat, y_bp, factorName = "BaselinePD"):
    
    if y_bp.ndim != 1:
        y_bp = y_bp.mean(axis=1)

    tmp = pd.DataFrame()
    tmp["sub"] =np.array(dat["sub"])
    tmp[factorName] = y_bp
     
    df_std = np.std(y_bp)
    df_ave = np.median(y_bp)
        
    df_std = df_std*3

    tmp["lower"] = df_ave - df_std
    tmp["upper"] = df_ave + df_std
        
    reject = tmp[(tmp[factorName] < tmp["lower"]) | (tmp[factorName] > tmp["upper"])]["sub"].values

    sns.histplot(data=tmp,x=factorName)
    plt.axvline(tmp["lower"][0],color="red")
    plt.axvline(tmp["upper"][0],color="red")

    return reject

#%% SDT
def SDT(hits, misses, fas, crs):
   
    """ returns a dict with d-prime measures given hits, misses, false alarms, and correct rejections"""
    
    if fas == 0:
       hit_rate = 1 - (1/(2*hits + misses))
       fa_rate = 1/(2*hits + misses)
       print("replaced!")

    else:
        # Floors an ceilings are replaced by half hits and half FA's
        half_hit = 0.5 / (hits + misses)
        half_fa  = 0.5 / (fas + crs)
     
        # Calculate hit_rate and avoid d' infinity
        hit_rate = hits / (hits + misses)
        if hit_rate == 1: 
            hit_rate = 1 - half_hit
        if hit_rate == 0: 
            hit_rate = half_hit
        
        # Calculate false alarm rate and avoid d' infinity
        fa_rate = fas / (fas + crs)
    
    
    # Return d', beta, c and Ad'
    out = pd.DataFrame()
    out['d'] = Z(hit_rate) - Z(fa_rate)
    out['beta'] = math.exp((Z(fa_rate)**2 - Z(hit_rate)**2) / 2)
    out['c'] = -(Z(hit_rate) + Z(fa_rate)) / 2
    out['Ad'] = norm.cdf(out['d'] / math.sqrt(2))
    
    return out

#%%
def MPCL(s, data_y, cfg, pcpd="PD", time_min = 0, time_max = 3,pltFlg=False):
    
    min_idx = []
    min_xval = []
    min_yval = []
    x = np.linspace(cfg["TIME_START"],cfg["TIME_END"],data_y.shape[1])
   
    mpcl = pd.DataFrame()
    mpcl["sub"] = np.unique(s)
    for mmName in ["idx","xval","yval"]:
        mpcl[mmName] = 0
    
    if pltFlg:
        plt.figure(figsize=(10,10))
    
    for i,iSub in enumerate(np.unique(s)):
        
        if isinstance(time_min, list):
            x_t = x[np.argwhere((x > time_min[i]) & (x <= time_max)).reshape(-1)]
            tmp_y = data_y[:,np.argwhere((x > time_min[i]) & (x <= time_max)).reshape(-1)]
        else:
            x_t = x[np.argwhere((x >= time_min) & (x <= time_max)).reshape(-1)]
            tmp_y = data_y[:,np.argwhere((x > time_min) & (x <= time_max)).reshape(-1)]
           
        ind = np.argwhere(s == iSub).reshape(-1)
         
        tmp_p = tmp_y[ind,:].copy()
        tmp_p = tmp_p.mean(axis=0)
         
        # second-order accurate central differences 
        pv = np.gradient(tmp_p)
        
        # find inflection point
        indices = np.where(np.diff(np.sign(pv)))[0]
        # plt.plot(x_t[indices],tmp_p[indices],'bo')  
        
        ev = []
        for itr in np.arange(len(indices)):
            if pv[indices[itr]] - pv[indices[itr]+1] > 0:
                ev.append(1)
            else:
                ev.append(0)
        
        if len(indices) > 0:  # peak pupil constriction/dilation(PC/PD)
            if pcpd == "PD":
                indices = indices[np.argwhere(np.array(ev) == 1).reshape(-1)]
            else:
                indices = indices[np.argwhere(np.array(ev) == 0).reshape(-1)]
        
            if len(indices) > 0: # if there is no PC/PD peak
                
                # PC/PD within 200ms will be rejected
                ini = 0
                rej = []
                for iPeak in np.arange(len(indices)):
                    if (indices[iPeak] - ini) < 0.2*cfg["SAMPLING_RATE"]:
                        rej.append(iPeak)
                    else:
                        ini = indices[iPeak]
                 
                indices = [d for i,d in enumerate(indices) if not i in rej]
         
        if pltFlg:
            plt.subplot(11,round(len(np.unique(s))/10),int(iSub))
            plt.plot(x_t,tmp_p,'-')
            
        if len(indices) > 0:  # peak pupil constriction/dilation(PC/PD)
            if pltFlg:
                plt.plot(x_t[indices],tmp_p[indices],'bo')
            
            if pcpd == "PD":
                indices = indices[np.argmax(tmp_p[indices])]
            else:
                indices = indices[np.argmin(tmp_p[indices])]
               
            mpcl["idx"][mpcl["sub"]==iSub] = int(getNearestValue(data_y[ind,:].mean(axis=0),tmp_p[indices]))
            mpcl["xval"][mpcl["sub"]==iSub] = float(x_t[indices])
            mpcl["yval"][mpcl["sub"]==iSub] = float(tmp_p[indices])
            ## min_xval.append(float(x_t[indices]))
            # min_yval.append(float(tmp_p[indices]))
            # min_idx.append(int(getNearestValue(data_y[ind,:].mean(axis=0),tmp_p[indices])))
            if pltFlg:
                plt.plot(x_t[indices],tmp_p[indices],'ro')
            
        else:
            if pcpd == "PD":
                mpcl["idx"][mpcl["sub"]==iSub] = int(getNearestValue(data_y[ind,:].mean(axis=0),tmp_p[0]))
                mpcl["xval"][mpcl["sub"]==iSub] = float(x_t[0])
                mpcl["yval"][mpcl["sub"]==iSub] = float(tmp_p[0])
               
                # plt.plot(x_t[0],tmp_p[0],'ro')
            else:
                mpcl["idx"][mpcl["sub"]==iSub] = int(getNearestValue(data_y[ind,:].mean(axis=0),tmp_p[-1]))
                mpcl["xval"][mpcl["sub"]==iSub] = float(x_t[-1])
                mpcl["yval"][mpcl["sub"]==iSub] = float(tmp_p[-1])
            
            if pltFlg:            
                plt.plot(x_t[-1],tmp_p[-1],'ro')
        
    return mpcl

