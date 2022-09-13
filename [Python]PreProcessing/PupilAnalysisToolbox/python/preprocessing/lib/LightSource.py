#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 12:19:11 2021

@author: yutasuzuki
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os
import json
from pre_processing import getNearestValue
import seaborn as sns        

class LightSource:    
    def __init__(self,cfg):
       self.cfg = cfg
       self.cfg['ipRGC'] = np.array(self.cfg['ipRGC'])*self.cfg['outPw']
       self.xyzTolms = [[0.3897,0.6890,0.0787],
                        [0.2298,1.1834,0.0464],
                        [0.0000,0.0000,1.0000]]

       self.XYZ_d65 = {'X':0.95047,
                       'Y':1,
                       'Z':1.08883}
       
       self.xyzToRGB = [[3.2410, -1.5374,-0.4986],
                        [-0.9692, 1.8760, 0.0416],
                        [0.0556, -0.2040, 1.0507]] 
       # self.xyzToRGB = [[0.412391,  0.357584,  0.180481],
       #                  [0.212639,  0.715169,  0.072192],
       #                  [0.019331,  0.119195,  0.950532]] 
       
    def getD65Spectra(self):
        ############ data loading (D65 and color matching function) ############
        Y_d65 = pd.read_csv(filepath_or_buffer="./data/spectrum_D65_1.csv", sep=",")
        Y_d65 = Y_d65[(Y_d65['lambda'] >= self.cfg['winlambda'][0]) & (Y_d65['lambda'] <= self.cfg['winlambda'][1])]
        
        self.Y_d65 = Y_d65
        return Y_d65
    
    def getXYZfunc(self):
        cl_func = pd.read_csv(filepath_or_buffer="./data/color-matching_function_wl-xyz.csv", sep=",")
        # cl_func = pd.read_csv(filepath_or_buffer="./data/whitelight/sbrgb10w.csv", sep=",")
        cl_func = cl_func[(cl_func['lambda'] >= self.cfg['winlambda'][0]) & (cl_func['lambda'] <= self.cfg['winlambda'][1])]
        self.cl_func = cl_func
        return  cl_func
   
    def getLMSfunc(self):
        lms = pd.read_csv(filepath_or_buffer="./data/linss2_10e_5.csv", encoding="ms932", sep=",")
        lms['S'][np.isnan(lms['S'])] = 0
        lms = lms[(lms['lambda'] >= self.cfg['winlambda'][0]) & (lms['lambda'] <= self.cfg['winlambda'][1])]
        self.lms = lms
        return lms
    
    def getipRGCfunc(self):
        ipRGC = pd.read_csv(filepath_or_buffer="./data/ipRGCSpectrum.csv", sep=",")

        ipRGC = ipRGC[(ipRGC['lambda'] >= self.cfg['winlambda'][0]) & (ipRGC['lambda'] <= self.cfg['winlambda'][1])]
        ipRGC['ipRGC'] = ipRGC['ipRGC'] /max(ipRGC['ipRGC'])
        self.ipRGC = ipRGC
        return ipRGC


    # def plotSpectrum(self):
    #     for ixyz,ilms,irgb in zip(['X','Y','Z'],['L','M','S'],['r','g','b']):
    #         plt.subplot(3,1,2); 
    #     plt.plot(cl_func['lambda'],cl_func[ixyz],irgb)
    
    #     plt.subplot(3,1,3); 
    #     plt.plot(lms['lambda'],lms[ilms],irgb)
    
    #     plt.plot(lms['lambda'],ipRGC['ipRGC'],'k')
    #     # plt.savefig("./color_matching_func.pdf")
    #     plt.show()
# plt.figure(figsize=(6,10))
# plt.subplot(3,1,1);plt.plot(Y_d65['lambda'],Y_d65['Y'])

    def getLEDspectra(self):
        
        ############ data loading (12LEDs embed in LED cube) Output power = 1000 (max) ############        
        dat_LED_spectrum = pd.DataFrame()
        plt.figure()
        for iLED in np.arange(1,self.cfg['numOfLEDs']+1):
            csv_input = pd.read_csv(filepath_or_buffer="./data/LED/LED" + str(iLED) + ".csv", encoding="ms932", sep=",")
            csv_input = csv_input.drop(['Unnamed: 1','Unnamed: 2','0'], axis=1)
        
            lam = csv_input.drop(np.arange(24), axis=0)
            dr = lam['メモ'].tolist()
            dr = [int(i[0:3]) for i in dr]
            dr = np.array(dr)
            
            csv_input = csv_input.set_index('メモ')
        
            dat_LED_spectrum['lambda'] = dr
            dat_LED_spectrum['LED' + str(iLED)] = np.array(np.float64(lam[str(self.cfg['outPw'])]))
            
            
            t = np.array([float(csv_input[str(self.cfg['maxoutPw'])]["X"]),
                          float(csv_input[str(self.cfg['maxoutPw'])]["Y"]),
                          float(csv_input[str(self.cfg['maxoutPw'])]["Z"])])
            t = t/max(t)
            
            rgb = np.round(np.dot(np.array(self.xyzToRGB),t)/255,4)
            rgb[rgb<0] = 0
            
            plt.plot(dat_LED_spectrum['lambda'], np.array(np.float64(lam[str(self.cfg['maxoutPw'])])),color=(rgb))
            # print(rgb)
            
            ind = np.argmax(np.array(np.float64(lam[str(self.cfg['maxoutPw'])])))
            plt.text(dat_LED_spectrum['lambda'][ind], np.array(np.float64(lam[str(self.cfg['maxoutPw'])]))[ind], "LED"+  str(iLED))
            
        # plt.savefig("./LEDs_power.pdf")
        # plt.show()
        
        dat_LED_spectrum = dat_LED_spectrum[dat_LED_spectrum['lambda'].values % 5 == 0]
        dat_LED_spectrum = dat_LED_spectrum[(dat_LED_spectrum['lambda'] >= self.cfg['winlambda'][0]) & (dat_LED_spectrum['lambda'] <= self.cfg['winlambda'][1])]
        
        self.dat_LED_spectrum = dat_LED_spectrum
        return dat_LED_spectrum

    def getXYZvalues(self,Y_d65,cl_func,dat_LED_spectrum,ipRGC):

        ############ XYZ values of 12LEDs ############
        k = 100/sum(Y_d65['Y'].values * cl_func['Y'].values)
        
        dat_LED_XYZ = {'X':[],'Y':[],'Z':[],'ipRGC':[]}
        for iLED in np.arange(1,self.cfg['numOfLEDs']+1):
            dat_LED_XYZ['X'].append(k * sum(dat_LED_spectrum['LED' + str(iLED)].values * cl_func['X'].values))
            dat_LED_XYZ['Y'].append(k * sum(dat_LED_spectrum['LED' + str(iLED)].values * cl_func['Y'].values))
            dat_LED_XYZ['Z'].append(k * sum(dat_LED_spectrum['LED' + str(iLED)].values * cl_func['Z'].values))
            dat_LED_XYZ['ipRGC'].append(sum(dat_LED_spectrum['LED' + str(iLED)].values * ipRGC['ipRGC'].values))
            
            # dat_LED_XYZ['X'].append(k * sum(dat_LED_spectrum['LED' + str(iLED)].values * lms['L'].values))
            # dat_LED_XYZ['Y'].append(k * sum(dat_LED_spectrum['LED' + str(iLED)].values * lms['M'].values))
            # dat_LED_XYZ['Z'].append(k * sum(dat_LED_spectrum['LED' + str(iLED)].values * lms['S'].values))
        return dat_LED_XYZ
    
            ############ XYZ values of 12LEDs ############
            
            # XYZ_d65 = {'X':sum(Y_d65['Y'].values * lms['L'].values),
            #             'Y':sum(Y_d65['Y'].values * lms['M'].values),
            #             'Z':sum(Y_d65['Y'].values * lms['S'].values)}
             
            # XYZ_d65['X'] = XYZ_d65['X'] / XYZ_d65['Y']
            # XYZ_d65['Z'] = XYZ_d65['Z'] / XYZ_d65['Y']
            # XYZ_d65['Y'] = XYZ_d65['Y'] / XYZ_d65['Y']
            
            # plt.plot((XYZ_d65['X']/sum(XYZ_d65.values())),(XYZ_d65['Y']/sum(XYZ_d65.values())),'o')
            
    def seekCombinations(self,dat_LED_XYZ,dat_LED_spectrum,cl_func,ipRGC):

        res = {'lambda':[],
               'LEDs':[],
               'spectrum':[],
               'coeff':[],
               'XYZ':[],'Yxy':[],'LMS':[],'ipRGC':[],
               
               'corrected_coeff':[],
               'corrected_LMS':[],'corrected_XYZ':[],'corrected_ipRGC':[],'corrected_Yxy':[],
               'corrected_spectrum':[]
               }
        
        ############ seek combination ############
        for iipRGC in self.cfg['ipRGC']:
            for comb in list(itertools.combinations(np.arange(self.cfg['numOfLEDs']),4)):
                dat = np.matrix([[dat_LED_XYZ['X'][comb[0]], dat_LED_XYZ['X'][comb[1]], dat_LED_XYZ['X'][comb[2]], dat_LED_XYZ['X'][comb[3]]],
                                 [dat_LED_XYZ['Y'][comb[0]], dat_LED_XYZ['Y'][comb[1]], dat_LED_XYZ['Y'][comb[2]], dat_LED_XYZ['Y'][comb[3]]],
                                 [dat_LED_XYZ['Z'][comb[0]] ,dat_LED_XYZ['Z'][comb[1]] ,dat_LED_XYZ['Z'][comb[2]], dat_LED_XYZ['Z'][comb[3]]],
                                 [dat_LED_XYZ['ipRGC'][comb[0]] ,dat_LED_XYZ['ipRGC'][comb[1]] ,dat_LED_XYZ['ipRGC'][comb[2]], dat_LED_XYZ['ipRGC'][comb[3]]]])
                
                dat_inv = dat**(-1)
            
                coeff = np.dot(np.array(dat_inv), np.array([self.XYZ_d65['X'],self.XYZ_d65['Y'],self.XYZ_d65['Z'],iipRGC])).reshape(-1)
            
                if not (any((x < 0 for x in coeff.tolist()))) | (any((x > self.cfg['maxoutPw'] for x in coeff.tolist()))):
                    spectrum_simulated_d65 = [dat_LED_spectrum['LED' + str(comb[0]+1)].values * (coeff[0]/self.cfg['outPw']),
                                              dat_LED_spectrum['LED' + str(comb[1]+1)].values * (coeff[1]/self.cfg['outPw']),
                                              dat_LED_spectrum['LED' + str(comb[2]+1)].values * (coeff[2]/self.cfg['outPw']),
                                              dat_LED_spectrum['LED' + str(comb[3]+1)].values * (coeff[3]/self.cfg['outPw'])]
                    
                    tmp = spectrum_simulated_d65.copy()
                    spectrum_simulated_d65 = np.sum(spectrum_simulated_d65,axis=0)
                          
                    if not any((x < 0 for x in spectrum_simulated_d65.tolist())):
                        
                        t = [sum(spectrum_simulated_d65 * cl_func['X'].values),
                             sum(spectrum_simulated_d65 * cl_func['Y'].values),
                             sum(spectrum_simulated_d65 * cl_func['Z'].values)]   
                    
                        res['LEDs'].append((np.array(comb)+1).tolist())
                        res['spectrum'].append(np.array(tmp).tolist())
                        res['lambda'].append(np.array(dat_LED_spectrum["lambda"]).tolist())
                        res['coeff'].append(np.round(coeff).tolist())
                        res['XYZ'].append(t)
                        res['Yxy'].append([t[1],(t[0]/sum(t)),(t[1]/sum(t))])
                        
                        res['LMS'].append(np.dot(np.array(self.xyzTolms),np.array(t)).tolist())
                        
                        # res['LMS'].append([sum(spectrum_simulated_d65 * lms['L'].values),
                        #                    sum(spectrum_simulated_d65 * lms['M'].values),
                        #                    sum(spectrum_simulated_d65 * lms['S'].values)])
                        
                        res['ipRGC'].append(sum(spectrum_simulated_d65 * ipRGC['ipRGC'].values))

        return res
    
    def validation(self,res):
     
        ################# validation using measured spectrum #################
        lut_spec = {}
        st = np.arange(1,1001)
        
        for iLED in  np.unique( res['LEDs']):
            
            csv_input = pd.read_csv(filepath_or_buffer="./data/LED/LED" + str(iLED) + ".csv", encoding="ms932", sep=",")
            csv_input = csv_input.drop(['Unnamed: 1','Unnamed: 2','0'], axis=1)
        
            csv_input = csv_input.drop(np.arange(24), axis=0)
            dr = csv_input['メモ'].tolist()
            x = np.arange(10,1010,10)
            
            lut_spec['LED' + str(iLED)] = []
            for ilambda in np.arange(len(dr)):
                
                y = np.float64(csv_input.values[ilambda,1:])
                a1, a2, b = np.polyfit(x, y, 2)
                
                lut_spec['LED' + str(iLED)].append(np.array(a1 * st**2 + a2 * st + b))
                
            lut_spec['LED' + str(iLED) + '_peak'] = np.max(np.array(lut_spec['LED' + str(iLED)]),axis=0)
            lut_spec['LED' + str(iLED)] = np.array(lut_spec['LED' + str(iLED)])
        
        ################# validation using measured spectrum #################
        
        for iLight,ipickedLED in zip(res['spectrum'],res['LEDs']):
            corrected_coeff = []
            for iLED,numLED in zip(iLight,ipickedLED):
                corrected_coeff.append(int(getNearestValue(lut_spec['LED' + str(numLED) + '_peak'],max(iLED))))
            res['corrected_coeff'].append(corrected_coeff)
        
        
        
        for i,(iLight,iCoeff) in enumerate(zip(res['LEDs'],res['corrected_coeff'])):
            spectrum_simulated_d65 = []
            for iLED,ic in zip(iLight,iCoeff):
                spectrum_simulated_d65.append(lut_spec['LED' + str(iLED)][:,int(ic)])
            
            spectrum_simulated_d65 = np.sum(spectrum_simulated_d65,axis=0)
            spectrum_simulated_d65 = spectrum_simulated_d65[(np.arange(380,781) >= self.cfg['winlambda'][0]) & (np.arange(380,781) <= self.cfg['winlambda'][1])]
        
            spectrum_simulated_d65 = spectrum_simulated_d65[np.arange(self.cfg['winlambda'][0],self.cfg['winlambda'][1]+1) % 5 == 0]
            
            t = [sum(spectrum_simulated_d65 * self.cl_func['X'].values),
                 sum(spectrum_simulated_d65 * self.cl_func['Y'].values),
                 sum(spectrum_simulated_d65 * self.cl_func['Z'].values)]
            # t = np.array(t) / t[1]
            res['corrected_spectrum'].append(spectrum_simulated_d65.tolist())
            res['corrected_XYZ'].append(t)
            res['corrected_Yxy'].append([t[1],(t[0]/sum(t)),(t[1]/sum(t))])
            
            res['corrected_LMS'].append(np.dot(np.array(self.xyzTolms),np.array(t)).tolist())
            
            # res['corrected_LMS'].append([sum(spectrum_simulated_d65 * lms['L'].values),
            #                              sum(spectrum_simulated_d65 * lms['M'].values),
            #                              sum(spectrum_simulated_d65 * lms['S'].values)])
            res['corrected_ipRGC'].append(sum(spectrum_simulated_d65 * self.ipRGC['ipRGC'].values))
    
        return res
    
    def rejectOutlier(self,res):

        ################# reject outlier #################
        
        tmp = np.array(res['corrected_Yxy'])[:,1:]
        sd_x = np.std(tmp[:,0])
        sd_y = np.std(tmp[:,1])
        
        rejectNum = (np.array(res['corrected_Yxy'])[:,1] > (np.mean(tmp[:,0]) + sd_x)) & (np.array(res['corrected_Yxy'])[:,1] < (np.mean(tmp[:,0]) - sd_x))
        
        for mm in res.keys():
            res[mm] = [d for i,d in enumerate(res[mm]) if not i in rejectNum]
        
        x = np.array(res["corrected_Yxy"])[:,1]
        y = np.array(res["corrected_Yxy"])[:,2]
       
        x_norm = x - x.mean()
        y_norm = y - y.mean()
        
        a = x_norm.std()**2
        b = y_norm.std()**2
        
        tmp_x = x_norm**2
        tmp_y = y_norm**2
        
        P = (tmp_x/a)+(tmp_y/b)-1
        
        rejectNum = np.argwhere(P > 0).reshape(-1)
        
        for mm in res.keys():
            res[mm] = [d for i,d in enumerate(res[mm]) if not i in rejectNum]
        
        
        y = np.array(res["corrected_Yxy"])[:,0]
    
        rejectNum = np.argwhere((y > (y.mean()+y.std()*0.5)) | (y < (y.mean()-y.std()*0.5)))
        
        for mm in res.keys():
            res[mm] = [d for i,d in enumerate(res[mm]) if not i in rejectNum]
        
        # y = np.array(res["corrected_LMS"])[:,2]
    
        # rejectNum = np.argwhere((y > (y.mean()+y.std())) | (y < (y.mean()-y.std())))
        
        # for mm in res.keys():
        #     res[mm] = [d for i,d in enumerate(res[mm]) if not i in rejectNum]
        
        return res
   
    
    def getMinMax(self,res):
         ################# choose min/max ipRGC #################
        
        filt = (res['ipRGC'] == max(res['ipRGC'])) | (res['ipRGC'] == min(res['ipRGC']))
        mmName = res.keys()
        for mm in mmName:
            res[mm] = [d for i,d in enumerate(res[mm]) if filt[i]]
        
        return res
    
    def plot(self,res,dat_LED_spectrum):
        
        ################# data plot #################
        
        plt.figure(figsize=(8,8))
        plt.style.use('ggplot')
        
        text = " L = {}, M = {}, \n S = {}, ipRGC = {}"
        
        for i,(sp,d_lms,d_Yxy) in enumerate(zip(res['spectrum'],res['LMS'],res['Yxy'])):
            
            plt.subplot(2, 3, 1)
            plt.plot(dat_LED_spectrum['lambda'],np.sum(np.array(sp),axis=0))
            # plt.ylim([0,0.02])
            plt.xlabel("wavelength")
            plt.ylabel("Power")
            
            # plt.text(500, 0.015+(i*0.003), 
            #           text.format(np.round(d_lms[0],4),
            #                       np.round(d_lms[1],4),
            #                       np.round(d_lms[2],4),
            #                       np.round(res['ipRGC'][i],4)),
            #                       fontsize=10)
            plt.subplot(2, 3, 2)
            plt.plot(d_Yxy[1],d_Yxy[2],'o')
            plt.xlim([0.28,0.34])
            plt.ylim([0.3,0.4])
            plt.xlabel("x")
            plt.ylabel("y")
        
        plt.subplot(2, 3, 3)
        plt.hist(d_Yxy[0])
        plt.hist(np.array(res['Yxy'])[:,0])
        # plt.xlim([0.28,0.34])
        # plt.ylim([0.3,0.4])
        plt.xlabel("x")
        plt.ylabel("y")
        
        for i,(sp,d_lms,d_Yxy) in enumerate(zip(res['corrected_spectrum'],res['corrected_LMS'],res['corrected_Yxy'])):
            
            plt.subplot(2, 3, 4)
            plt.plot(dat_LED_spectrum['lambda'],sp)
            # plt.ylim([0,0.02])
            plt.xlabel("wavelength")
            plt.ylabel("Power")
            
            # plt.text(500, 0.015+(i*0.003), 
            #           text.format(np.round(d_lms[0],4),
            #                       np.round(d_lms[1],4),
            #                       np.round(d_lms[2],4),
            #                       np.round(res['corrected_ipRGC'][i],4)),
            #                       fontsize=10)  
            plt.subplot(2, 3, 5)
            plt.plot(d_Yxy[1],d_Yxy[2],'o')
            plt.xlim([0.28,0.34])
            plt.ylim([0.3,0.4])
            plt.xlabel("x")
            plt.ylabel("y")
            
        plt.subplot(2, 3, 6)
        plt.hist(np.array(res['corrected_Yxy'])[:,0])
        # plt.xlim([0.28,0.34])
        # plt.ylim([0.3,0.4])
        plt.xlabel("x")
        plt.ylabel("y")
        # plt.savefig("./calcuratedOut.pdf")
        
        # plt.savefig("res.eps")
        plt.show()
        

    def saveData(self,res):
    
        with open(os.path.join("./data_LEDCube.json"),"w") as f:
                json.dump(res,f)



def showSpectra(filePath,saveFig=False):
    
   
    # for iLED in np.arange(1,self.cfg['numOfLEDs']+1):
    csv_input = pd.read_csv(filepath_or_buffer = filePath , encoding="ms932", sep=",")
    csv_input = csv_input.drop(['基準','白色板','レシオ分母'], axis=1)
    spectra = csv_input.drop(np.arange(24), axis=0)
    csv_input = csv_input.set_index('Unnamed: 0')
    
    lam = [int(t[:3]) for t in spectra["Unnamed: 0"].values]
  
    spectra = spectra.set_index('Unnamed: 0')
    spectra = np.array(spectra.astype(float, errors = 'raise')).T

    dr = csv_input.loc['メモ'].tolist()
   
    dat_spectrum = pd.DataFrame()
    dat_spectrum["y"] = spectra.reshape(-1)
    dat_spectrum["data_x"] = np.tile(lam, spectra.shape[0])
    dat_spectrum["condition"] = np.repeat(dr,spectra.shape[1])
    dat_spectrum["Yxy_x"] = np.repeat(np.array(csv_input.loc['x'].astype(float, errors = 'raise')),spectra.shape[1])
    dat_spectrum["Yxy_y"] = np.repeat(np.array(csv_input.loc['y'].astype(float, errors = 'raise')),spectra.shape[1])
    dat_spectrum["Yxy_Y"] = np.repeat(np.array(csv_input.loc['Y'].astype(float, errors = 'raise')),spectra.shape[1])
    
    # dat_spectrum["x"] = np.round(dat_spectrum["x"],3)
    # dat_spectrum["y"] = np.round(dat_spectrum["y"],3)
    
    #%%
    plt.figure()
    grid = sns.FacetGrid(dat_spectrum, col="condition", col_wrap=2, size=5)
    grid.map(sns.lineplot, "data_x", 'y', ci='none')
    plt.xlabel("lambda[nm]")
    plt.ylabel("radiance[W・sr-1・m-2]")
    
    if saveFig:
        plt.savefig("spectra_sep.pdf")
        
    #%%
    plt.figure()
    sns.lineplot(data=dat_spectrum, x="data_x", y='y', hue="condition", ci='none')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.xlabel("lambda[nm]")
    plt.ylabel("radiance[W・sr-1・m-2]")
    if saveFig:
        plt.savefig("spectra.pdf")

    #%%
    tmp_plot = dat_spectrum.groupby(["condition"],as_index=False).agg(np.nanmean)

    plt.figure()
    # sns.pointplot(data = tmp_plot, x="Yxy_x", y='Yxy_y', hue="condition", ci='none')
    for x,y,l in zip(tmp_plot["Yxy_x"].values,tmp_plot["Yxy_y"].values,tmp_plot["condition"].values):
        plt.plot(x,y,'o',label = l)
    
    plt.legend()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.xlim(0.3, 0.36)
    plt.ylim(0.3, 0.4)
    
    plt.xlabel("x")
    plt.ylabel("y")
    
    if saveFig:
        plt.savefig("xyY.pdf")

    #%% Y
    plt.figure()
    ax = sns.pointplot(data=dat_spectrum, x="condition", y='Yxy_Y',group = "condition")
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
    if saveFig:
        plt.savefig("xyY_Y.pdf")

    return dat_spectrum


def showLEDSpectra(numLEDs,pw):
    
    dat_LED_spectrum = pd.DataFrame()

   
    csv_input = pd.read_csv(filepath_or_buffer="./data/LED/LED" + str(numLEDs) + ".csv", encoding="ms932", sep=",")
    csv_input = csv_input.drop(['Unnamed: 1','Unnamed: 2','0'], axis=1)

    lam = csv_input.drop(np.arange(24), axis=0)
    dr = lam['メモ'].tolist()
    dr = [int(i[0:3]) for i in dr]
    dr = np.array(dr)
    
    csv_input = csv_input.set_index('メモ')

    dat_LED_spectrum['lambda'] = dr
    dat_LED_spectrum['LED' + str(numLEDs)] = np.array(np.float64(lam[str(pw)]))
    
    plt.plot(dat_LED_spectrum['lambda'], np.array(np.float64(lam[str(pw)])))
    
    ind = np.argmax(np.array(np.float64(lam[str(pw)])))
    plt.text(dat_LED_spectrum['lambda'][ind], np.array(np.float64(lam[str(pw)]))[ind], "LED"+  str(numLEDs))
        
   