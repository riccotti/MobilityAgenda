import math

from collections import defaultdict

from kmeans import *

__author__ = 'Riccardo Guidotti'


def euclidean_distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def points_labels_to_clusters(points, labels):
    clusters = defaultdict(list)
    for i in range(0, len(points)):
        clusters[labels[i]].append(points[i])
    return clusters


def clusters_to_points_labels(clusters):
    points = list()
    labels = list()
    for c, pl in clusters.iteritems():
        for p in pl:
            points.append(p)
            labels.append(c)
    return points, labels


def get_clusters_centers(clusters, dist):
    centers = list()
    for c, pl in clusters.iteritems():
        centers.append(get_center(pl, dist))

    return centers


def get_center(points, dist):
    center = None
    min_sse = float('infinity')

    for p1 in points:
        sse = 0
        for p2 in points:
            d = dist(p1, p2)
            sse += d
        if sse < min_sse:
            min_sse = sse
            center = p1

    return center


class BisectiveKmeans:

    def __init__(self, eps, distances=euclidean_distances, distance=euclidean_distance, verbose=False):
        self.eps = eps
        self.distances = distances
        self.distance = distance
        self.verbose = verbose

        self.k_ = -1
        self.data_ = None
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, data):
        bisecting_center_cluster = self._kmeans_clustering(data)

        while True:
            new_bisecting_center_cluster = dict()
            for c, pl in bisecting_center_cluster.iteritems():
                if self._to_be_bisected(c, pl):
                    bcc = self._kmeans_clustering(pl)
                    for c_bcc, pl_bcc in bcc.iteritems():
                        new_bisecting_center_cluster[c_bcc] = pl_bcc
                else:
                    new_bisecting_center_cluster[c] = pl
            if len(new_bisecting_center_cluster) == len(bisecting_center_cluster):
                break
            bisecting_center_cluster = new_bisecting_center_cluster

        k = len(bisecting_center_cluster)

        labels = list()
        points = list()
        centers = list()
        cid = 0
        for c, pl in bisecting_center_cluster.iteritems():
            for p in pl:
                points.append(p)
                labels.append(cid)
            cid += 1
            centers.append(c)

        self.k_ = k
        self.data_ = points
        self.labels_ = labels
        self.cluster_centers_ = centers

        return self

    def _kmeans_clustering(self, data):
        kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
        kmeans.fit(data, distance_function=self.distances)
        labels = kmeans.labels_
        clusters = points_labels_to_clusters(data, labels)
        centers = get_clusters_centers(clusters, self.distance)
        bisecting_center_cluster = dict()
        for cid, pl in clusters.iteritems():
            bisecting_center_cluster[(centers[cid][0], centers[cid][1])] = pl
        return bisecting_center_cluster

    def _to_be_bisected(self, c, pl):
        if len(pl) == 1:
            return False
        for p in pl:
            d = self.distance(c, p)
            if d > self.eps:
                return True
        return False
