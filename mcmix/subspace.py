import numpy as np
from numba import jit, njit, prange
from tqdm import tqdm
import tensorflow as tf

## ALGORITHM: SUBSPACE ESTIMATION


#Function to estimate h, array of empirical next state probabilities given state and action
@njit(parallel=True, cache=True)
def geth(onehotsa, onehotsp, simple=False):
    h = np.zeros((onehotsa.shape[0], onehotsa.shape[2], onehotsa.shape[3], onehotsa.shape[2])) #m, s, a, sp
    N_msa = np.zeros((onehotsa.shape[0], onehotsa.shape[2], onehotsa.shape[3]))
    for m in range(onehotsa.shape[0]):
        for s in range(onehotsa.shape[2]):
            for a in range(onehotsa.shape[3]):
                for sp in range(onehotsa.shape[2]):
                    for t in range(onehotsa.shape[1]):
                        h[m,s,a,sp] += onehotsa[m,t,s,a]*onehotsp[m,t,sp]
                        N_msa[m,s,a] += onehotsa[m,t,s,a]
    if not simple:
        for m in range(onehotsa.shape[0]):
            for s in range(onehotsa.shape[2]):
                for a in range(onehotsa.shape[3]):
                    for sp in range(onehotsa.shape[2]):
                        if N_msa[m,s,a] != 0:
                            h[m,s,a,sp] /= N_msa[m,s,a]
                        else:
                            h[m,s,a,sp] = 0
    else:
        h /= onehotsa.shape[1]
    return h

#function to get projections of next state probabilities to rank K subspaces
def getEig(onehotsa, onehotsp, omegaone, omegatwo, K, wt = True, smalldata=True, device='/CPU:0'):
    #h1 and h2 are shaped (m,s,a,s')
    h1 = np.array(geth(onehotsa[:,omegaone,:,:], onehotsp[:,omegaone,:]), dtype=np.float32)
    h2 = np.array(geth(onehotsa[:,omegatwo,:,:], onehotsp[:,omegatwo,:]), dtype=np.float32)
    
    #Hsa = (h1 * h2).sum(3).mean(0)
    #Hsa = h1[:,:,:,:,None] * h2[:,:,:,None,:]
    #Hsa = np.einsum('ijkl,ijkm->ijklm', h1, h2).mean(0) #somehow einsum is faster? but equivalent
    
    nStates = h1.shape[1]
    nActions = h1.shape[2]
    
    if not wt:
        invwts = np.ones((nStates, nActions))
    else:
        #trajwts is shaped (s,a)
        trajwts = (onehotsa[:,omegaone,:,:].sum(axis=1) * onehotsa[:,omegatwo,:,:].sum(axis=1)).sum(0)
        invwts = 1/trajwts
        (invwts)[np.isinf(invwts)] = 0
    if smalldata:
        Hsa = ((h1[...,None] @ h2[...,None,:])*invwts[None,:,:,None,None]).sum(0)
    else:
        with tf.device(device):
            Hsa = tf.zeros((nStates, nActions, nStates, nStates), dtype=tf.float32)
            #Hsa = [[tf.zeros((nStates, nStates), dtype=tf.float32) for a in range(nActions)] for s in range(nStates)]
            for m in tqdm(range(len(h1))):
                Hsa += (tf.convert_to_tensor(h1[m,:,:,:,None], np.float32) 
                        @ tf.convert_to_tensor(h2[m,:,:,None,:], np.float32))*invwts[:,:,None,None]
                #for s in range(nStates):
                #    for a in range(nActions):
                #        Hsa[s][a] += (tf.convert_to_tensor(h1[m,s,a,:,None], np.float32) 
                #                     @ tf.convert_to_tensor(h2[m,s,a,None,:], np.float32))*invwts[s,a]
            
            Hsa = Hsa.numpy()/len(h1)
            #for s in range(nStates):
            #    for a in range(nActions):
            #        Hsa[s][a] = Hsa[s][a].numpy()/len(h1)
    Hsa = np.array(Hsa)
    Hsa = Hsa + Hsa.transpose(0,1,3,2)
    eigvalsa, eigvecsa = np.linalg.eigh(Hsa)
    return eigvalsa[:,:,-K:], eigvecsa[:,:,:,-K:]

#function to get projections of occupancy measures to rank K subspaces
def getEigKs(onehotsa, onehotsp, omegaone, omegatwo, K):
    k1 = onehotsp[:,omegaone,:].mean(1)
    k2 = onehotsp[:,omegatwo,:].mean(1)
    Ks = (k1[...,None] @ k2[...,None,:]).mean(0)
    eigvalsp, eigvecsp = np.linalg.eigh(Ks + Ks.T)
    return eigvalsp[-K:], eigvecsp[:,-K:]

#helper function to get estimates of h, 
#  array of empirical next state probabilities given state and action,
#  for lists of indexes of each partition of \Omega_1 and \Omega_2
def geths(onehotsa, onehotsp, omgones, omgtwos, G):
    hs = []
    for g in tqdm(range(G)):
        hs.append([geth(onehotsa[:,omgones[g],:,:], onehotsp[:,omgones[g],:]), 
                   geth(onehotsa[:,omgtwos[g],:,:], onehotsp[:,omgtwos[g],:])])
    return np.array(hs)


