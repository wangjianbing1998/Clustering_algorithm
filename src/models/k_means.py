from src.utils import *


def computeSSE(data, centers, clusterID):

    sse = 0
    nData = len(data)
    for i in range(nData):
        c = clusterID[i]
        sse += squaredDistance(data[i], centers[c])

    return sse


def updateClusterID(data, centers):

    nData = len(data)
    nCenters = len(centers)

    clusterID = [0] * nData
    dis_Centers = [0] * nCenters

    for i in range(nData):
        for c in range(nCenters):
            dis_Centers[c] = squaredDistance(data[i], centers[c])
        clusterID[i] = dis_Centers.index(min(dis_Centers))
    return clusterID


def updateCenters(data, clusterID, K):
    nDim = len(data[0])
    centers = [[0] * nDim for i in range(K)]

    ids = sorted(set(clusterID))
    for id in ids:
        indices = [i for i, j in enumerate(clusterID) if j == id]
        cluster = [data[i] for i in indices]
        if len(cluster) == 0:
            centers[id] = [0] * nDim
        else:
            centers[id] = [float(sum(col)) / len(col) for col in zip(*cluster)]
    return centers


def run_kmeans(data, centers, maxIter=100, tol=1e-6):

    nData = len(data)

    if nData == 0:
        return []

    K = len(centers)

    clusterID = [0] * nData

    if K >= nData:
        for i in range(nData):
            clusterID[i] = i
        return clusterID

    nDim = len(data[0])

    lastDistance = 1e100

    for iter in range(maxIter):
        clusterID = updateClusterID(data, centers)
        centers = updateCenters(data, clusterID, K)

        curDistance = computeSSE(data, centers, clusterID)
        if lastDistance - curDistance < tol or (lastDistance - curDistance) / lastDistance < tol:
            # print(("# of iterations:", iter))
            # print(("SSE = ", curDistance))
            return clusterID

        lastDistance = curDistance

    # print(("# of iterations:", iter))
    # print(("SSE = ", curDistance))
    return clusterID


def kmeans(args, data):
    import numpy as np

    K = args.nb_clusters
    centers = []
    for i in range(K):
        centers.append(data[i])
    results = run_kmeans(data, centers)
    return np.array(results)
