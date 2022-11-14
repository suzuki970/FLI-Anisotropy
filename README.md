# Experimental data for *"The lateralized flash-lag illusion: A psychophysical and pupillometry study"*
Copyright 2022 Yuta Suzuki


### Article information
Yuta Suzuki*, Sumeyya Atmaca, Bruno Laeng. The lateralized flash-lag illusion: A psychophysical and pupillometry study, in review.

## Requirements
Python
- pre-processing (https://github.com/suzuki970/PupilAnalysisToolbox)
- numpy
- scipy
- os
- json

R
- library(rjson)
- library(ggplot2)
- library(ggpubr)
- library(Cairo)
- library(gridExtra)
- library(effsize)
- library(BayesFactor)
- library(rjson)
- library(reshape)
- library(lme4)
- library(permutes)

## Raw data
raw data can be found at **'[Python]PreProcessing/results'**

## Pre-processing
- Raw data (.asc) are pre-processed by **'[Python]PreProcessing/parseData.py'**

	- Pre- processed data is saved as **‘data_original.json’**

- Artifact rejection and data epoch are performed by **'[Python]PreProcessing/dataAnalysis_eyeMetrics.py'** and **'[Python]PreProcessing/dataAnalysis_task.py'**


## Figure and statistics
-  **'[Rmd]Results/figure.Rmd'**‘ is to generate figures and statistical results.
