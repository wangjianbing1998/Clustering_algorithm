from math import exp

from src.models.k_means import run_kmeans
from src.utils import *


def kernel(data, sigma):
    """
    RBF kernel-k-means
    :param data: data points: list of list [[a,b],[c,d]....]
    :param sigma: Gaussian radial basis function
    :return:
    """
    nData = len(data)
    Gram = [[0] * nData for i in range(nData)]  # nData x nData matrix
    # TODO
    # Calculate the Gram matrix

    # symmetric matrix
    for i in range(nData):
        for j in range(i, nData):
            if i != j:  # diagonal element of matrix = 0
                # RBF kernel: K(xi,xj) = e ( (-|xi-xj|**2) / (2sigma**2)
                square_dist = squaredDistance(data[i], data[j])
                base = 2.0 * sigma ** 2
                Gram[i][j] = exp(-square_dist / base)
                Gram[j][i] = Gram[i][j]
    return Gram


def kmeans_kernel(args, data):
    data = kernel(data, args.sigma)

    K = args.nb_clusters

    centers = []
    for i in range(K):
        centers.append(data[i])

    results = run_kmeans(data, centers)

    return results
