import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sklearn
import matplotlib.pyplot as plt
import scipy
plt.style.use('matplotlibrc')

## ALGORITHM: CLUSTERING

## TEST STATISTIC CALCULATION
'''
def computeStat(h1, h2, eigvecsa, device='/CPU:0'):
    with tf.device(device):
        h1t = tf.convert_to_tensor(h1, np.float64)
        h2t = tf.convert_to_tensor(h2, np.float64)
        eigvecsat = tf.convert_to_tensor(eigvecsa, np.float64)
        proj1t = tf.squeeze(h1t[..., None,:] @ eigvecsat[None,...])
        proj2t = tf.squeeze(h2t[..., None,:] @ eigvecsat[None,...])
        statmns = tf.reduce_max(
                        tf.reduce_sum(
                            (proj1t[None,...] - proj1t[:,None,...]) * 
                            (proj2t[None,...] - proj2t[:,None,...]), 
                        axis=-1),
                    axis=(2,3)).numpy()
    return statmns
'''

def computeStat(hs, eigvecsa, numpy=True, smalldata=True, device='/CPU:0', proj=True):
    if numpy:
        if proj:
            #2, nTraj, S, A, 1, K
            projs = (hs[..., None,:] @ eigvecsa[None,...]).squeeze(4)
        else:
            projs = hs
        if smalldata:
            statmns = np.max(
                        np.nansum(
                            (projs[0,None,...] - projs[0,:,None,...]) * 
                            (projs[1,None,...] - projs[1,:,None,...]), 
                        axis=-1), 
                    axis=(2,3))
        else:
            statmns = np.zeros((projs.shape[1], projs.shape[1]))
            for s in tqdm(range(projs.shape[2])):
                for a in range(projs.shape[3]):
                    newstats = np.nansum((projs[0, None, :, s, a, :] - projs[0, :, None, s, a, :]) * 
                                (projs[1, None, :, s, a, :] - projs[1, :, None, s, a, :]), -1)
                    statmns = np.maximum(statmns, 
                       newstats)
        return statmns
    else:
        with tf.device(device):
            hs = tf.convert_to_tensor(hs, np.float32)
            eigvecsa = tf.convert_to_tensor(eigvecsa, np.float32)
            if proj:
                projs = tf.squeeze(hs[..., None,:] @ eigvecsa[None,...], axis=4)
            else:
                projs = hs
            if smalldata:
                statmns = tf.reduce_max(
                                    tf.reduce_sum(
                                        (projs[0,None,...] - projs[0,:,None,...]) * 
                                        (projs[1,None,...] - projs[1,:,None,...]), 
                                    axis=-1), 
                            axis=(2,3)).numpy()
            else:
                statmns = tf.zeros((projs.shape[1], projs.shape[1]), dtype=tf.float32)
                for s in tqdm(range(projs.shape[2])):
                    for a in range(projs.shape[3]):
                        newstats = tf.reduce_sum((projs[0, None, :, s, a, :] - projs[0, :, None, s, a, :]) * 
                                    (projs[1, None, :, s, a, :] - projs[1, :, None, s, a, :]), -1)
                        statmns = tf.math.maximum(statmns, newstats)
                statmns = statmns.numpy()
        return statmns

## OBTAINING CLUSTERS
def getClusters(statmns, thresh, K, method='kmeans'):
    return sklearn.cluster.spectral_clustering((statmns < thresh).astype(int), n_clusters=K,
                                                         assign_labels='kmeans')


## Hungarian algorithm to obtain permuted accuracy
def getAcc(clusters, labels, K):
    costs = np.array([[np.mean((clusters==i).astype(int) != 
                           (labels==j).astype(int)) for j in range(K)] 
                  for i in range(K)])
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(costs)
    return np.mean(np.array([col_ind[int(i)] for i in clusters]) == labels)

## DIAGNOSTICS
def clusterDiagnostics(statmns, K, labels, lo, hi, step, method='kmeans', figsize=(16,9), trials=1):
    accs = []
    wts = []
    taus = np.arange(lo, hi, step)
    for trial in range(trials):
        accTrial = []
        wtTrial = []
        for tau in tqdm(taus):
            clusterlabs = getClusters(statmns, thresh=tau, K=K, method=method)
            accTrial.append(getAcc(clusterlabs, labels, K))
            if K == 2:
                wtTrial.append(max(np.mean(clusterlabs==1), np.mean(clusterlabs==0)))
        accs.append(accTrial)
        if K == 2:
            wts.append(wtTrial)
    plt.figure(figsize=figsize)
    plt.plot(taus, 100*np.array(accs).mean(0), label='Accuracies (%)')
    if K == 2:
        plt.plot(taus, 100*np.array(wts).mean(0), label='Max. Cluster Weight (%)')
    plt.xlabel('Threshold')
    plt.ylabel('Clustering Accuracy')
    plt.title('Accuracy Against Thresholds')
    plt.legend()
    