from multiprocessing.pool import Pool
from subprg import InputData,DispResults

from mcmcFunc import scale_sgm,mcmcFunc

#
#       Yasuharu Okamoto, 2016.11
#

def main():

    X, n, meanSt, sgmSt, fout, nm_fout = InputData()
    
    # tmp_df = df.copy()
    # g1 = [-83.3,-66.7,-50.0,-33.3,-16.7,0,16.7,33.3]     
    # tmp_df["Condition"] = [np.round(g1[int(i-1)],3) for i in df["Condition"].values.tolist()]
    
    # tmp_df = tmp_df[(tmp_df["sub"]==1) &
    #                 (tmp_df["vField"]=="LVF")]
    # X = [ [c,t+1] for (c,t) in zip(tmp_df["Condition"],tmp_df["Task"])] 
    # n = len(X)
    # meanSt = tmp_df["Condition"].values.mean()
    
    # out = pd.DataFrame()
    # out["ID"] = np.arange(n)+1
    # out["St"] = tmp_df["Condition"]
    # out["Res"] = tmp_df["Task"]+1
    # out.to_csv("new.txt", sep=" ", index = False)
    
    NSimu = 1000  
    
    params = {'jump' : 0,
              'n' : n,
              'X' : X,
              'NSimu' : NSimu,
              'initMu' : meanSt,
              'initSgm' : sgmSt,
              'genSgmMu' : sgmSt/10.0,
              'genSgmSgm' : sgmSt/10.0}
    
    #
    #       Adjust the parameters in MCMC
    #
    
    istep = 0
    while True:
        print("\nStep-{} started.".format(istep))
        
        Mu, Sgm, cnt_mu, cnt_sgm = mcmcFunc( params )
    
        sum_mu = 0.0
        sum_sgm = 0.0
        for t in range(1, NSimu + 1):
            sum_mu += Mu[t]
            sum_sgm += Sgm[t]
        params['initMu'] = sum_mu / NSimu
        params['initSgm'] = sum_sgm / NSimu
        # print("meanMu = {}".format(params['initMu']))
        # print("meanSgm = {}".format(params['initSgm']))
                
        ck_cnt = 0
        r_mu = cnt_mu / NSimu
        # print("r_mu = {}".format(r_mu))
        v, ck_cnt = scale_sgm( r_mu, ck_cnt )
        params['genSgmMu'] = params['genSgmMu'] * v
        # print("genSgmMu = {}".format(params['genSgmMu']))
        r_sgm = cnt_sgm / NSimu;
        # print("r_sgm = {}".format(r_sgm))
        v, ck_cnt = scale_sgm( r_sgm, ck_cnt )
        params['genSgmSgm'] = params['genSgmSgm'] * v
        # print("genSgmSgm = {}".format(params['genSgmSgm']))
        # print("ck_cnt = {}".format(ck_cnt))
        if ck_cnt == 0:
            break
        istep += 1
    
    # #
    # #           The main MCMCs
    # #
    NSimu = 1000
    params['NSimu'] = NSimu
    n_mcmc = 4
    ary_params = []
    for i in range(0, n_mcmc):
        temp = params.copy()
        temp['jump'] = i + 1
        ary_params.append(temp)
    
    # print("\nThe main MCMCs started.")
    pool = Pool()
    results = pool.map(mcmcFunc, ary_params)    #   Multiprocessing

    # print("\nThe main MCMCs ended.\n")
    mrg_mu = []
    mrg_sgm = []
    for temp_Mu, temp_Sgm, cnt_mu, cnt_sgm in results:
        mrg_mu += temp_Mu[1:NSimu + 1]
        mrg_sgm += temp_Sgm[1:NSimu + 1]
    # print("Length(mrg_mu) = {}".format(len(mrg_mu)))
    # print("Length(mrg_sgm) = {}".format(len(mrg_sgm)))
    mrg_mu.sort()
    mrg_sgm.sort()
    
    # DispResults( NSimu, n_mcmc, X, n, mrg_mu, mrg_sgm, fout)


if __name__ == "__main__":
    main()
    

# fout.close()
# print("\nOutput file {} was saved.".format(nm_fout))

