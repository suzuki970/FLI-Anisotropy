import math
import matplotlib.pyplot as plt
# import numpy as np
from RNGen import RNp2to191
from common import cum_normalKC

def InputData():
    
    nm_fin = input("Input data file = ")
    fin = open(nm_fin, 'rt')

    nm_fout = input('Output file name = ')
    fout = open(nm_fout, 'wt')
    fout.write("Input data file...{}\n".format(nm_fin))

    data = fin.readlines()
    fin.close()

    pos = 0
    while True:
        if len(data[pos]) > 0:
            if data[pos][0] == '/':
                break
        pos += 1

    ID = []
    X = []
    pos += 1
    while True:
        if data[pos][0] == '/':
            break
        temp = data[pos].split()
        print(temp)
        ID.append(temp[0])
        tempL = []
        tempL.append(float(temp[1]))
        tempL.append(int(temp[2]))
        X.append(tempL)
        pos += 1

    n = len(X)
    for i in range(0, n):
        fout.write("{0:>5}   {1}    {2}\n".format(ID[i], X[i][0], X[i][1]))

    sumSt = 0.0
    for i in range(0, n):
        sumSt += X[i][0]
    meanSt = sumSt / n
    ssumSt = 0.0
    for i in range(0, n):
        ssumSt += (X[i][0] - meanSt) ** 2.0
    varSt = ssumSt / n
    sgmSt = math.sqrt(varSt)

    return X, n, meanSt, sgmSt, fout, nm_fout


def DispResults( NSimu, n_mcmc, X, n, mrg_mu, mrg_sgm):
    Q1Mu = mrg_mu[int(NSimu * n_mcmc * 0.25)]
    MedMu = mrg_mu[int(NSimu * n_mcmc / 2.0)]
    Q3Mu = mrg_mu[int(NSimu * n_mcmc * 0.75)]
    LMu = mrg_mu[int(NSimu * n_mcmc * 0.025)]
    UMu = mrg_mu[int(NSimu * n_mcmc * 0.975)]
    Q1Sgm = mrg_sgm[int(NSimu * n_mcmc * 0.25)]
    MedSgm = mrg_sgm[int(NSimu * n_mcmc / 2.0)]
    Q3Sgm = mrg_sgm[int(NSimu * n_mcmc * 0.75)]
    LSgm = mrg_sgm[int(NSimu * n_mcmc * 0.025)]
    USgm = mrg_sgm[int(NSimu * n_mcmc * 0.975)]
    print("Med. of Mu = {}".format(MedMu))
    print("Med. of Sgm = {}".format(MedSgm))
    # fout.write("\nMed. of PSE = {0:.5}\n".format(MedMu))
    # fout.write("Q1(PSE) = {0:.5}    Q3(PSE) = {1:.5}\n".format(Q1Mu, Q3Mu))
    # fout.write("95% CI(PSE) = [{0:.5}, {1:.5}]\n".format(LMu, UMu))
    # fout.write("\nMed. of JND = {0:.5}\n".format(MedSgm * 0.67449))
    # fout.write("Q1(JND) = {0:.5}    Q3(JND) = {1:.5}\n".format(Q1Sgm * 0.67449, Q3Sgm * 0.67449))
    # fout.write("95% CI(JND) = [{0:.5}, {1:.5}]\n".format(LSgm * 0.67449, USgm * 0.67449))
    
    min_St = X[0][0]
    max_St = X[0][0]
    for i in range(1, n):
        if min_St > X[i][0]:
            min_St = X[i][0]
        if max_St < X[i][0]:
            max_St = X[i][0]
    min_x = min_St - 0.05 * (max_St - min_St)
    max_x = max_St + 0.05 * (max_St - min_St)
  
    plt.axis([min_x, max_x, 0.0, 1.0])

    phy = []
    psy = []
    v = min_x
    while v <= max_x:
        phy.append(v)
        psy.append(cum_normalKC((v - MedMu) / MedSgm))
        v += 0.01 * (max_x - min_x)

    rn = RNp2to191()
    p_st = []
    p_res = []
    for i in range(0, n):
        p_st.append(X[i][0])
        if X[i][1] == 1:
            p_res.append(0.1 * rn.uni())
        else:
            p_res.append(1.0 - 0.1 * rn.uni())
                                
    plt.title("Psychometric function   PSE = {0:.5}   JND = {1:.5}".format(MedMu, MedSgm * 0.67449))
    plt.plot(phy, psy, 'b', linewidth = 5)
    plt.plot(p_st, p_res, 'go')
    plt.legend(['PM', 'Data'], loc = 2)
    plt.show()

    plt.title("PSE (Med. = {0:.5})".format(MedMu))
    plt.hist(mrg_mu, bins = 20)
    plt.show()

    mrg_JND = []
    for v in mrg_sgm:
        mrg_JND.append(v * 0.67449)
    plt.title("JND (Med. = {0:.5})".format(MedSgm * 0.67449))
    plt.hist(mrg_JND, bins = 20)
    plt.show()
    


