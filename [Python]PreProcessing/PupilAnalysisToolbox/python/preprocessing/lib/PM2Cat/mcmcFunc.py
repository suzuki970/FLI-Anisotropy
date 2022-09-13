import math
from RNGen import RNp2to191
from common import cum_normalKC


def sqr( v ):
    return v * v

def MyLog( x ):
    v = 0.0
    if (x > 1.0e-308):
        v = math.log(x)
    else:
        v = math.log(1.0e-308)
    return v

def MyExp( x ):
    v = 1.0
    if (x < 709.0):
        v = math.exp(x)
    else:
        v = math.exp(709.0)
    return v

def exp0( x ):
    if (x > 0.0):
        return 1.0
    else:
        return math.exp(x)

def LogL( X, n, mu, sgm ):
    sum = 0.0
    for i in range(0, n):
        if X[i][1] == 2:
            sum += MyLog( cum_normalKC((X[i][0] - mu) / sgm) )
        else:
            sum += MyLog( 1.0 - cum_normalKC((X[i][0] - mu) / sgm) )
    return sum

def scale_sgm( acpt_r, ck ):
    v = 1.0
    if (acpt_r > 0.5):
        ck += 1
        v = 1.0 + 2.0 * (acpt_r - 0.4) / 0.6
    elif (acpt_r < 0.3):
        ck += 1
        v = 1.0 / (1.0 + 2.0 * (0.4 - acpt_r) / 0.4)
    return v, ck
    



def mcmcFunc( params ):
    n_jump = params['jump']
    n = params['n']
    tempX = params['X']
    X = []
    for data in tempX:
        X.append(data)
    NSimu = params['NSimu']
    Mu = []
    Mu.append(params['initMu'])
    Sgm = []
    Sgm.append(params['initSgm'])
    genSgmMu = params['genSgmMu']
    genSgmSgm = params['genSgmSgm']
    cnt_mu = 0
    cnt_sgm = 0

    rn = RNp2to191()
    for i in range(0, n_jump):
        rn.jump(100)
              
    for t in range(0, NSimu):
        if (t % 500) == 0:
            print("{}/{}...jump = {}".format(t, NSimu, n_jump))
        
        y = rn.normalMS(Mu[t], genSgmMu)
        num = LogL( X, n, y, Sgm[t] )
        den = LogL( X, n, Mu[t], Sgm[t] )
        a = exp0(num - den)
        if rn.uni() < a:
            Mu.append(y)
            cnt_mu += 1
        else:
            Mu.append(Mu[t])

        y = rn.normalMS(Sgm[t], genSgmSgm)
        if y <= 0.0:
            a = -1.0
        else:
            num = LogL( X, n, Mu[t + 1], y )
            den = LogL( X, n, Mu[t + 1], Sgm[t] )
            a = exp0(num - den)
        if rn.uni() < a:
            Sgm.append(y)
            cnt_sgm += 1
        else:
            Sgm.append(Sgm[t])

    return Mu, Sgm, cnt_mu, cnt_sgm

            



