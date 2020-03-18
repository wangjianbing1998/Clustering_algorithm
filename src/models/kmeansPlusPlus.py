# coding=gbk
import math
import random

import numpy as np

from sklearn.cluster import KMeans


class K_Means_Plus_Plus:

    def __init__(self, points_list, k):
        self.centroid_count = 0
        self.point_count = len(points_list)
        self.cluster_count = k
        self.points_list = list(points_list)
        self.initialize_random_centroid()
        self.initialize_other_centroids()

        self.pp = list(points_list)

    def initialize_random_centroid(self):
        self.centroid_list = []
        index = random.randint(0, len(self.points_list) - 1)

        self.centroid_list.append(self.remove_point(index))
        self.centroid_count = 1

    def remove_point(self, index):
        new_centroid = self.points_list[index]
        del self.points_list[index]

        return new_centroid

    """Finds the other k-1 centroids from the remaining lists of points"""

    def initialize_other_centroids(self):
        while not self.is_finished():
            distances = self.find_smallest_distances()
            chosen_index = self.choose_weighted(distances)
            self.centroid_list.append(self.remove_point(chosen_index))
            self.centroid_count += 1

    def find_smallest_distances(self):
        distance_list = []

        for point in self.points_list:
            distance_list.append(self.find_nearest_centroid(point))

        return distance_list

    def find_nearest_centroid(self, point):
        min_distance = math.inf

        for values in self.centroid_list:
            distance = self.euclidean_distance(values, point)
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def choose_weighted(self, distance_list):
        distance_list = [x ** 2 for x in distance_list]
        weighted_list = self.weight_values(distance_list)
        indices = [i for i in range(len(distance_list))]
        return np.random.choice(indices, p=weighted_list)

    def weight_values(self, list):
        sum = np.sum(list)
        return [x / sum for x in list]

    def euclidean_distance(self, point1, point2):
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)

        return np.linalg.norm(point2 - point1)

    def is_finished(self):
        outcome = False
        if self.centroid_count == self.cluster_count:
            outcome = True

        return outcome

    def final_centroids(self):
        return self.centroid_list

    def get_result(self, nb_clusters):

        labels = []
        for p in self.pp:
            dis = float('inf')
            i = 0
            for index, c in enumerate(self.centroid_list):
                d = self.euclidean_distance(p, c)
                if d < dis:
                    dis = d
                    i = index
            labels.append(i)
        labels = KMeans(nb_clusters).fit_predict(self.pp)
        return labels


def kmeansPP(args, data):
    k = K_Means_Plus_Plus(data, args.nb_clusters)
    return k.get_result(args.nb_clusters)
