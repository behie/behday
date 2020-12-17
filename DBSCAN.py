import numpy as np


import numpy
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def MyDBSCAN(D, eps, MinPts):

    labels = [0]*len(D)
    C = 0
    for P in range(0, len(D)):
        if not (labels[P] == 0):
           continue
        NeighborPts = regionQuery(D, P, eps)
        if len(NeighborPts) < MinPts:
            labels[P] = -1  
        else: 
           C += 1
           growCluster(D, labels, P, NeighborPts, C, eps, MinPts)
    

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    lab = numpy.array(labels)
    print(lab)
    # plt.scatter(lab[:,0], lab[:,1])
    plt.show()

def growCluster(D, labels, P, NeighborPts, C, eps, MinPts):

    labels[P] = C
    i = 0
    while i < len(NeighborPts):           
        Pn = NeighborPts[i]
        if labels[Pn] == -1:
           labels[Pn] = C
        elif labels[Pn] == 0:
            labels[Pn] = C
            PnNeighborPts = regionQuery(D, Pn, eps)
            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts
        i += 1        

def regionQuery(D, P, eps):

    neighbors = []
    for Pn in range(0, len(D)):
        if numpy.linalg.norm(D[P] - D[Pn]) < eps:
           neighbors.append(Pn)
            
    return neighbors



import pandas as pd
X = pd.read_csv("clustering.csv")
X = StandardScaler().fit_transform(X)


MyDBSCAN(X,0.3,5)


headers = ["x","y"]
X = pd.read_csv("clustering.csv", names = headers)
X.head()

plt.scatter(X.x, X.y)
plt.show()