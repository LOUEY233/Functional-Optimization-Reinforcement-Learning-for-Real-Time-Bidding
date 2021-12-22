import math
import numpy as np
import pandas as pd

batch_size = 20
gamma = 0.95

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 20000


def epsilon_by_frame(request):
    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * request / epsilon_decay)

def win_prob_second_list(bi,wi,zi):
    #print(data)
    bk = list(set(bi))
    dj = list()
    nj = list()
    wo = list()
    for i in bk:
        # loss = len(data[(data["zi"] < i-1) & (data["wi"]==1)]) + len(data[(data["bi"] <= i-1) & (data["wi"]==0)])
        nj.append(sum([(z >= i-1) & (w == 1) for z,w in zip(zi, wi)]) + sum([(b >= i) & (w == 0) for b,w in zip(bi, wi)]))
        dj.append(sum([(z == i-1) & (z > 0) for z in zi]))
        # nj.append(8 - loss)
    wo =[1]
    for c,i,j,k in zip(range(len(bk)),bk,nj,dj):
        if c>0:
             wo.append(((j-k)/(j+0.000001)) * wo[c-1])
        else:
            continue
    wo = list(map(lambda x:1-x,wo))
    return bk,wo

if __name__ == '__main__':
    print(epsilon_by_frame(150000))
