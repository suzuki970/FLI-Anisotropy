
data_tertile = data.frame()
data_pupilAll = data.frame()
gaze_data = data.frame()
gaze_all = data.frame()
angle_count = data.frame()
ind_data = data.frame()
minLatency = data.frame()
data_pupilAll_res = data.frame()

folderName = list.files(rootFolder, full.names=T)
folderName = folderName[length(folderName)]

for(ivField in c("LVF","RVF")){
  
  # data loading --------------------------------------------------------------------
  f = list.files(folderName, pattern=paste0("data",ivField,normName), full.names=T)
  dat = fromJSON(file=f)
  for(iName in 1:(length(dat)-1)){
    dat[[iName]] = unlist(dat[[iName]])
  }
  dat$numOfTrial = 1:length(dat$sub)
  # pupil ---------------------------------------------------------------
  dataVF = makePupilDataset(dat,
                            c("PDR","zeroArray"),
                            c("SOA",'Task','RT'),
                            c(cfg$TIME_START,cfg$TIME_END),
                            list(NULL,NULL,NULL),
                            list(NULL,NULL,NULL))
  
  dataVF$vField = ivField
  data_pupilAll = rbind(data_pupilAll,dataVF)
  
  # pupil res ------------------------------------------------------------
  dataVF_res = makePupilDataset(dat,
                            c("PDR_res"),
                            c("SOA",'Task','RT','numOfTrial'),
                            c(-1,-cfg$TIME_START+cfg$TIME_END),
                            list(NULL,NULL,NULL,NULL),
                            list(NULL,NULL,NULL,NULL))
  
  dataVF_res$vField = ivField
  data_pupilAll_res = rbind(data_pupilAll_res,dataVF_res)
  
  # saccades ---------------------------------------------------------------
  tmp_angle_count = makePupilDataset(dat,
                                     c("angle_count"),
                                     c("SOA",'Task'),
                                     c(0,359),
                                     list(NULL,NULL),
                                     list(NULL,NULL))
  tmp_angle_count$vField = ivField
  angle_count <- rbind(angle_count,tmp_angle_count)
  
  # task ---------------------------------------------------------------
  
  for(iName in names(dat)){
    if (length(dat[[iName]]) != length(dat$sub) ){
      eval(parse(text=paste0("dat$",iName," =NULL")))
    }
  }
  
  tmp_ind_data = as.data.frame(dat)
  tmp_ind_data$vField = ivField
  ind_data <- rbind(ind_data,tmp_ind_data)
  
  # eye-fixation --------------------------------------------------------------------
  
  f = list.files(folderName, pattern=paste0("fix_all",ivField), full.names=T)
  tmp_gaze_all = fromJSON(file=f)
  for(iName in 1:length(tmp_gaze_all)){
    tmp_gaze_all[[iName]] = unlist(tmp_gaze_all[[iName]])
  }
  tmp_gaze_all = as.data.frame(tmp_gaze_all)
  tmp_gaze_all$vField = ivField
  
  tmp_gaze_all[tmp_gaze_all$vField == "LVF",]$x = -tmp_gaze_all[tmp_gaze_all$vField == "LVF",]$x
  
  tmp_gaze_all$x = atanh(((tmp_gaze_all$x*cfg$DOT_PITCH)/10)/cfg$VISUAL_DISTANCE) * (180/pi)
  tmp_gaze_all$y = atanh(((tmp_gaze_all$y*cfg$DOT_PITCH)/10)/cfg$VISUAL_DISTANCE) * (180/pi)
  
  reject = setdiff(unique(tmp_gaze_all$sub),unique(tmp_ind_data$sub))
  for (iSub in reject){
    tmp_gaze_all = tmp_gaze_all[tmp_gaze_all$sub != iSub,]
  }
  
  gaze_all <- rbind(gaze_all,tmp_gaze_all)
  
}


gaze_hist = data.frame()
for(ivField in unique(gaze_all$vField)){
  for(iSub in unique(gaze_all[gaze_all$vField == ivField,]$sub)){
    
    tmp = hist(gaze_all[gaze_all$vField == ivField &
                          gaze_all$sub == iSub,]$x,
               breaks = seq(-30,30,length=100),
               freq = FALSE, plot = FALSE)
    
    t = gaze_all[gaze_all$sub == iSub &
                   gaze_all$vField == ivField,]$x
    
    fit <- fitdist(t[t>0], distr = "gamma", method = "mle")
    
    # bfit1 <- bootdist(fit,bootmethod="param",niter=1000)
    
    gaze_hist = rbind(gaze_hist,data.frame(
      sub = iSub,
      vField = ivField,
      breaks = tmp$breaks[2:length(tmp$breaks)],
      shape = summary(fit)$estimate["shape"],
      rate = summary(fit)$estimate["rate"],
      density = tmp$density
    ))
  }
}

#### to be "ahead" as 1 in the RVF
ind_data[ind_data$vField == "RVF",]$Task = 1 - ind_data[ind_data$vField == "RVF",]$Task
data_pupilAll[data_pupilAll$vField == "RVF",]$Task = 1 - data_pupilAll[data_pupilAll$vField == "RVF",]$Task
data_pupilAll_res[data_pupilAll_res$vField == "RVF",]$Task = 1 - data_pupilAll_res[data_pupilAll_res$vField == "RVF",]$Task

#### RT correction in each SOA 
ind_data$RT = ind_data$RT + 250 - ind_data$SOA
data_pupilAll$RT = data_pupilAll$RT + 250 - data_pupilAll$SOA

# task data load --------------------------------------------------------------------

f = list.files(folderName, pattern=paste0("data_df",normName), full.names=T)
df = fromJSON(file=f)
for(iName in 1:length(df)){
  df[[iName]] = unlist(df[[iName]])
}

df = as.data.frame(df)

f = list.files(folderName, pattern=paste0("velocity",normName), full.names=T)
data_velocity = fromJSON(file=f)
for(iName in 1:length(data_velocity)){
  data_velocity[[iName]] = unlist(data_velocity[[iName]])
}

data_velocity = as.data.frame(data_velocity)

minLatency  = data.frame(
  sub = unique(df$sub),
  max = cfg$maxVal,
  min = cfg$minVal,
  ave_max_y=0,
  ave_min_y=0
)

f = list.files(folderName, pattern=paste0("data_task",normName), full.names=T)
data_task = fromJSON(file=f)

for(iName in 1:length(data_task)){
  data_task[[iName]] = unlist(data_task[[iName]])
}

ind_task_data = as.data.frame(data_task)
ind_task_data$RT = ind_task_data$RT + 250 - data_task$SOA

ind_task_data[ind_task_data$vField == "RVF",]$Task = 1 - ind_task_data[ind_task_data$vField == "RVF",]$Task

ind_data$numOfTrial=0

for(ivField in unique(ind_data$vField)){
  for(iSub in sort(unique(ind_data[ind_data$vField == ivField,]$sub))){
    tmp = ind_data[ ind_data$sub == iSub &
                      ind_data$vField == ivField,]
    ind_data[ ind_data$sub == iSub &
                ind_data$vField == ivField,]$numOfTrial = 1:dim(tmp)[1]
  }
}

# aggregation -------------------------------------------------------------
data_e1 = aggregate( . ~ sub*SOA*vField, data = ind_data, FUN = "mean")
data_task = ind_data
data_pupil = aggregate( . ~ sub*data_x*SOA*vField, data = data_pupilAll, FUN = "mean")
data_pupil_res = aggregate( . ~ sub*data_x*SOA*vField, data = data_pupilAll_res, FUN = "mean")


# PSE (each sub) ---------------------------------------------------------

psyCurves = data.frame()
psyCurves_ind = data.frame()
dat_th_ind = data.frame()

ind_task_data$n = 1
tmp = aggregate( . ~ sub*SOA*vField, data = ind_task_data, FUN = "sum")
fit_vField_sub = quickpsy(tmp, SOA, Task, n, grouping = c("sub", "vField"), bootstrap = "none")

psyCurves_ind = data.frame(
  sub = fit_vField_sub$curves$sub,
  vField = fit_vField_sub$curves$vField,
  x = fit_vField_sub$curves$x,
  y = fit_vField_sub$curves$y
)
dat_th_ind = data.frame(
  sub = fit_vField_sub$thresholds$sub,
  vField = fit_vField_sub$thresholds$vField,
  th = fit_vField_sub$thresholds$thre
)

# reject data (due to PSE estimation failed) -------------------------------------------------------------
eFlag = TRUE
reject = NULL
dumy_dat_th_ind = dat_th_ind
rejectAll = NULL
tmp_zscored = abs(scale(dumy_dat_th_ind$th))
reject = sort(unique(dumy_dat_th_ind[tmp_zscored > median(tmp_zscored)*1.5,]$sub))

if (length(reject) == 0){
  eFlag = FALSE
}
for (iSub in reject){
  for(valName in c("dumy_dat_th_ind")){
    eval(parse(text=paste0(valName,"=",valName,"[",valName,"$sub != iSub,]")))
  }
}

mu = mean(dumy_dat_th_ind$th)
sigma = sd(dumy_dat_th_ind$th)*3

tmp = unique(dumy_dat_th_ind[dumy_dat_th_ind$th > mu+sigma |
                               dumy_dat_th_ind$th < mu-sigma,]$sub)

reject = sort(cbind(t(reject),t(tmp)))

##### if estimated threshold is too large(>300ms)
data_e1_keep = data_e1
psyCurves_ind_keep = psyCurves_ind

filelib = c("ind_data","dat_th_ind","data_tertile",
            "gaze_data","gaze_all","gaze_hist",
            "data_e1","psyCurves_ind",
            "data_pupilAll","data_pupilAll_res",
            "data_pupil","data_pupil_res",
            "gaze_data","angle_count","minLatency","data_velocity")

for (iSub in reject){
  for(valName in filelib){
    eval(parse(text=paste0(valName,"=",valName,"[",valName,"$sub != iSub,]")))
  }
  data_e1_keep[data_e1_keep$sub == iSub,]$sub = paste0(data_e1_keep[data_e1_keep$sub == iSub,]$sub,"_rejected")
  psyCurves_ind_keep[psyCurves_ind_keep$sub == iSub,]$sub = paste0(psyCurves_ind_keep[psyCurves_ind_keep$sub == iSub,]$sub,"_rejected")
}

print(paste0("rejected sub = ",reject))

reject = NULL
for(iSub in sort(unique(data_tertile$sub))){
  for(ivField in unique(data_tertile$vField)){
    for(iLag in sort(unique(data_tertile$SOA))){
      d = data_tertile[data_tertile$sub == iSub &
                         data_tertile$vField == ivField &
                         data_tertile$SOA == iLag,] 
      # print(dim(d))
      if(dim(d)[1] == 0) {
        reject = cbind(reject,iSub)
      }
    }
  }
}
reject = unique(sort(reject))
for (iSub in reject){
  for(valName in filelib){
    eval(parse(text=paste0(valName,"=",valName,"[",valName,"$sub != iSub,]")))
  }
  data_e1_keep[data_e1_keep$sub == iSub,]$sub = paste0(data_e1_keep[data_e1_keep$sub == iSub,]$sub,"_rejected")
  psyCurves_ind_keep[psyCurves_ind_keep$sub == iSub,]$sub = paste0(psyCurves_ind_keep[psyCurves_ind_keep$sub == iSub,]$sub,"_rejected")
}


##### if either of VF is rejected
reject = setdiff(ind_data$sub, df$sub)
print(paste0("rejected sub = ",reject))

for (iSub in reject){
  for(valName in filelib){
    eval(parse(text=paste0(valName,"=",valName,"[",valName,"$sub != iSub,]")))
  }
  data_e1_keep[data_e1_keep$sub == iSub,]$sub = paste0(data_e1_keep[data_e1_keep$sub == iSub,]$sub,"_rejected")
  psyCurves_ind_keep[psyCurves_ind_keep$sub == iSub,]$sub = paste0(psyCurves_ind_keep[psyCurves_ind_keep$sub == iSub,]$sub,"_rejected")
}

# velocity -------------------------------------------------------------

data_pupil_timeCourse =  aggregate( PDR ~ sub*data_x*vField, data = data_pupil, FUN = "mean")
data_pupil_timeCourse = data_pupil_timeCourse[data_pupil_timeCourse$data_x >= 0 & data_pupil_timeCourse$data_x <= eTime,]

for(iSub in sort(unique(data_pupil_timeCourse$sub))){
  tmp = data_pupil_timeCourse[data_pupil_timeCourse$sub == iSub,]
  tmp =  aggregate( PDR ~ sub*data_x, data = tmp, FUN = "mean")
  
  a = mean(minLatency[minLatency$sub == iSub,]$max)
  max_idx = unique(which(abs(tmp$data_x - a) ==  min(abs(tmp$data_x - a))))[1]
  minLatency[minLatency$sub == iSub,]$ave_max_y = tmp$PDR[max_idx]
  
  a = mean(minLatency[minLatency$sub == iSub,]$min)
  min_idx = unique(which(abs(tmp$data_x - a) ==  min(abs(tmp$data_x - a))))[1]
  minLatency[minLatency$sub == iSub,]$ave_min_y = tmp$PDR[min_idx]
}
 
# save data -------------------------------------------------------------

for(valName in filelib){
  eval(parse(text=paste0(valName,"$sub=subName[",valName,"$sub]")))
}

if(!dir.exists(dataFolder)){
  dir.create(dataFolder)
}

data_e1_keep$sub = factor(data_e1_keep$sub,unique(data_e1_keep$sub))
psyCurves_ind_keep$sub = factor(psyCurves_ind_keep$sub,unique(psyCurves_ind_keep$sub))

save(ind_data,
     data_pupil,data_pupil_res,
     data_pupilAll,data_pupilAll_res,
     minLatency,data_velocity,
     gaze_data,gaze_all,gaze_hist,
     angle_count,ind_task_data,
     data_e1,data_tertile,data_e1_keep,psyCurves_ind_keep,
     psyCurves_ind,dat_th_ind,
     file = paste0(dataFolder,"dataset_task",normName,".rda"))

# PSE(average) ---------------------------------------------------------
tmp = ind_task_data
tmp$sub = NULL
tmp$n = 1
tmp = aggregate( . ~ SOA*vField, data = tmp, FUN = "sum")
fit_VFs = quickpsy(tmp, SOA, Task, n, grouping = c("vField"), B = 2000)

psyCurves = data.frame(
  sub = "Average",
  vField = fit_VFs$curves$vField,
  x = fit_VFs$curves$x,
  y = fit_VFs$curves$y
)

dat_th = data.frame(
  sub = "Average",
  vField = fit_VFs$thresholds$vField,
  th = fit_VFs$thresholds$thre
)

save(fit_vField_sub,fit_VFs,
     psyCurves,dat_th,
     file = paste0(dataFolder,"dataset_task_bootstrap.rda"))
