import numpy as np
from numba import jit, njit, prange
import multiprocessing
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import copy
import helpers
from clustering import *

# Gets starting state probabilities per mixture component
def getStartWeights(states, predlabs, K, nStates, hard=True):
    wts = np.zeros((K, nStates))
    if hard:
        for i in range(len(states)):
            wts[predlabs[i],states[i, 0]] += 1
        return wts/np.array([sum(predlabs==k) for k in range(K)])[:,None]
    else:
        for i in range(len(states)):
            for k in range(K):
                if np.isnan(predlabs[k, i]):
                    wts[k,states[i, 0]] += 1/wts.shape[1]
                else:
                    wts[k,states[i, 0]] += predlabs[k, i]
                #wts[k,states[i, 0]] += predlabs[k, i]
        #print(wts)
        return wts/np.nansum(wts, axis=1)[:,None]

@njit(parallel=False, cache=True)
def getPolicyHelperSoft(states, actions, K, nStates, nActions, predprobs):
    pi_ksa = np.zeros((K, nStates, nActions)) #k, s, a
    for i in range(states.shape[0]):
        for j in range(states.shape[1]):
            for k in range(K):
                pi_ksa[int(k), 
                       int(states[i,j]), 
                       int(actions[i,j])] += predprobs[k, i]
    return pi_ksa

@njit(parallel=False, cache=True)
def getPolicyHelperLabs(states, actions, K, nStates, nActions, predlabs):
    pi_ksa = np.zeros((K, nStates, nActions)) #k, s, a
    for i in range(states.shape[0]):
        for j in range(states.shape[1]):
            pi_ksa[int(predlabs[i]), 
                   int(states[i,j]), 
                   int(actions[i,j])] += 1
    return pi_ksa

@njit(parallel=False, cache=True)
def getPolicyHelper(states, actions, nStates, nActions):
    pi_sa = np.zeros((nStates, nActions)) #s,a
    for i in range(states.shape[0]):
        for j in range(states.shape[1]):
            pi_sa[int(states[i,j]), int(actions[i,j])] += 1
    return pi_sa

def getPolicyEstim(states, actions, K, nStates, nActions, preds=None, hard=True):
    if not hard:
        pi = getPolicyHelperSoft(states, actions, K, nStates, nActions, preds)
    elif preds is not None:
        pi = getPolicyHelperLabs(states, actions, K, nStates, nActions, preds)
    else:
        pi = getPolicyHelper(states, actions, nStates, nActions)
    pi = pi/np.nansum(pi, axis=-1)[...,None]
    pi[np.isnan(pi)] = 1/nActions
    return pi

@njit(parallel=False, cache=True)
def getmodelestim(clusterlabs, states, actions, nextstates,
                  K, nStates, nActions):
    Phat_ksa = np.zeros((K, nStates, nActions, nStates))
    for i in prange(states.shape[0]):
        for j in range(states.shape[1]):
            Phat_ksa[int(clusterlabs[i]), 
                     int(states[i,j]), 
                     int(actions[i,j]), 
                     int(nextstates[i,j])] += 1
    return Phat_ksa

@njit(parallel=False, cache=True)
def getmodelestimsoft(expect, states, actions, nextstates,
                  K, nStates, nActions):
    Phat_ksa = np.zeros((K, nStates, nActions, nStates))
    for i in prange(states.shape[0]):
        for j in range(states.shape[1]):
            for k in range(K):
                Phat_ksa[k, 
                         int(states[i,j]), 
                         int(actions[i,j]), 
                         int(nextstates[i,j])] += expect[k,i]
    return Phat_ksa

def getModelEstim(clusterlabs, states, actions, nextstates,
                  K, nStates, nActions, hard=True):
    if hard:
        model = getmodelestim(clusterlabs, states, actions, nextstates,
                  K, nStates, nActions)
    else:
        model = getmodelestimsoft(clusterlabs, states, actions, nextstates,
                  K, nStates, nActions)
    model = model / np.nansum(model, axis=-1)[..., None]
    model[np.isnan(model)] = 1/nStates
    return model


def getloglik(expect, modelestim, states, actions, nextstates, hard=True):
    if hard:
        return np.nansum(
                    np.nansum(
                        np.log(modelestim[:, states, actions, nextstates]), 
                    axis=-1)[expect, np.arange(len(states))])
    else:
        return np.nansum(
                    np.nansum(
                        np.log(np.nansum(modelestim[:, states, actions, nextstates] 
                                        * expect[...,None], axis=0)),
                    axis=-1))

#Classification of new trajectories with the hard and soft E-step
def classify(model, states, actions, nextstates, policy=None, reg=0, prior=None, startweights=None, labs=True):
    if policy is not None:
        probs = np.nansum(np.log(model[:, states, actions, nextstates])
                    + np.log(policy[:, states, actions]), axis=-1)
    else:
        probs = np.nansum(np.log(model[:, states, actions, nextstates]), axis=-1)
    if startweights is not None:
        probs += np.log(startweights[:,states[:,0]])
    
    probs += np.random.uniform(high=1e-7, size=probs.shape)
    if reg > 0:
        probs += reg*np.log(prior)[:,None]
    if labs:
        return probs.argmax(0)
    else:
        return probs
    
#Classification of new trajectories with Algorithm 3
#Not the E-step, here we classify by minimum projected distance
#    to the K models estimated per mixture element
#This seems to have better single-step theoretical guarantees
#    than one E-step but in practice it seems the E-step does better
#Recommend you use classify() instead
######INPUTS######
#dataclust: dataset of s, a, k, r, s' on clustering partition
#clusterlabs: cluster labels of len(dataclust)
#hsubs: \hat{P}_{n,i}(\cdot|s,a), estimated models per trajectory in subspace partition
#Phat_ksa: model estimated from clustering dataset post-clustering
##################
def classifyProj(dataclust, clusterlabs, hsubs, Phat_ksa, K, nStates, nActions):
    #estimating frequency of k,s,a given s,a
    freq_ksa = (np.array([
                        helpers.getN_sa(dataclust[clusterlabs==k],
                              nStates,nActions,reshape=True) 
                          for k in range(K)])
                /helpers.getN_sa(dataclust,nStates,nActions,
                         reshape=True)[None,...])
    
    #forming matrix to eigendecompose
    Mclust_sa = (freq_ksa[...,None,None] * 
                 (Phat_ksa[...,None] @ Phat_ksa[...,None,:])).sum(0)
    Mclust_sa = Mclust_sa + Mclust_sa.transpose(0,1,3,2)
    
    #eigendecomposition, get top K eigenvectors
    eigvalclustsa, eigvecclustsa = np.linalg.eigh(Mclust_sa)
    eigvalclust = eigvalclustsa[:,:,-K:]
    eigvecclust = eigvecclustsa[:,:,:,-K:]
    
    #projection of each model to eigenspaces
    projmod = (Phat_ksa[...,None,:] @ eigvecclust[None,...]).squeeze() #k, s, a, embed
    #projection of each trajectory-wise-estimated-model to eigenspaces
    projsub = (hsubs[..., None,:] @ eigvecclust[None,...]).squeeze() #i, n, s, a, embed
    
    #inner product for 'distances', max over s,a pairs, take argmax over mixture elements
    projclass = (((projsub[0,None,...] - projmod[:,None,...]) * 
                        (projsub[1,None,...] - projmod[:,None,...])).sum(-1))
    projclass += np.random.uniform(high=1e-9, size=projclass.shape)
    projclass = projclass.max((2,3)).argmax(0)
    return projclass
    
def em(expect, modelestim, states, actions, nextstates, labels, 
                K, nStates, nActions, prior, 
               max_iter = 100, min_iter = 10, checkin=5, reg = 0,
               permute=False, permutation=0, verbose=True, hard=True):
    i = 0
    modelold = np.ones(modelestim.shape)
    while i < min_iter or np.nansum(np.abs(modelold - modelestim)) > 1e-3:
        modelold = modelestim
        
        if hard:
            policy = getPolicyEstim(states, actions, 
                         K, nStates, nActions, expect, hard=True)
            modelestim = getModelEstim(expect.astype(int), states, actions, nextstates,
                                       K=K, nStates=nStates, nActions=nActions, hard=True)
            startweights = getStartWeights(states, expect, K, nStates, hard=True)
            expectprobs = np.nansum(np.log(modelestim[:, states, actions, nextstates])
                                    + np.log(policy[:, states, actions]), -1) + np.log(startweights[:,states[:,0]])
            expectprobs += np.random.uniform(high=1e-7, size=expectprobs.shape)
            expect = (expectprobs + #random number to perturb argmax 
                      reg*np.log(prior)[:,None]).argmax(0)
            prior = np.bincount(np.concatenate([expect,
                                               np.arange(K)]
                                    ))/(len(expect)+K)
        else:
            policy = getPolicyEstim(states, actions, 
                         K, nStates, nActions, expect, hard=False)
            modelestim = getModelEstim(expect, states, actions, nextstates,
                                       K=K, nStates=nStates, nActions=nActions, hard=False)
            startweights = getStartWeights(states, expect, K, nStates, hard=False)
            expectprobs = (np.nansum(np.log(modelestim[:, states, actions, nextstates]) 
                                    + np.log(policy[:, states, actions]), axis=-1) 
                           + np.log(startweights[:,states[:,0]]))
            expectprobs += np.random.uniform(high=1e-7, size=expectprobs.shape)
            expectprobs += reg*np.log(prior)[:,None]
            expect = np.exp(expectprobs - np.max(expectprobs))# + np.random.uniform(high=1e-7, size =expect.shape)
            expect = (expect / np.nansum(np.abs(expect), axis=0))
            prior = np.bincount(np.concatenate([expect.argmax(0),
                                               np.arange(K)]
                                    ))/(len(expect)+K) #fix to soft prior
        
        i += 1
        if i % checkin == 0 and verbose:
            print('iteration', i, 'diff', np.nansum(np.abs(modelold - modelestim)))
            expectlabs = expect if hard else expect.argmax(0)
            print(len(expectlabs), len(labels))
            if permute:
                if K == 2:
                    print('permuted accuracy:', max(np.mean(expectlabs == labels), np.mean(expectlabs != labels)))
                else:
                    print('permuted accuracy:', getAcc(expectlabs, labels, K))
                    
            else:
                if K == 2:
                    print('accuracy:', [np.mean(expectlabs == labels), np.mean(expectlabs != labels)][permutation])
                else:
                    #have to print permuted, cant enumerate all perms
                    print('permuted accuracy:', getAcc(expectlabs, labels, K)) 
            print(getloglik(expect, modelestim, states, actions, nextstates, hard=hard))
        if i > max_iter:
            break
    loglik = getloglik(expect, modelestim, states, actions, nextstates, hard=hard)
    if verbose:
        print('log-likelihood:', loglik)
    return expect, modelestim, loglik



