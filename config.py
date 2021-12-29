import math
import numpy as np
import pandas as pd

batch_size = 20
gamma = 0.95

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 3000


def epsilon_by_frame(request):
    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * request / epsilon_decay)

def win_prob_second_list(bi,wi,zi):
    bk = list(set(bi))
    dj = list()
    nj = list()
    wo = list()
    for i in bk:
        # loss = len(data[(data["zi"] < i-1) & (data["wi"]==1)]) + len(data[(data["bi"] <= i-1) & (data["wi"]==0)])
        nj.append(np.sum(np.logical_and((np.array(zi) >= (i-1)),np.array(wi) == 1)) + np.sum(np.logical_and((np.array(bi) >= i),np.array(wi) == 0)))
        dj.append(np.sum(np.logical_and((np.array(zi) == (i-1)),np.array(zi) > 0)))
        # nj.append(8 - loss)
    wo =[1]
    for c,i,j,k in zip(range(len(bk)),bk,nj,dj):
        if c>0:
             wo.append(((j-k)/(j+0.0000001)) * wo[c-1])
    wo = list(map(lambda x:1-x,wo))
    return bk,wo

if __name__ == '__main__':
    print(epsilon_by_frame(3000))
