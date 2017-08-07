import warnings
from geohash import *
from scipy import stats
from scipy.special import erfc
from collections import defaultdict
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

from myagenda.xmeans import *


__author__ = 'Riccardo Guidotti'


warnings.filterwarnings('ignore', category=DeprecationWarning)


def chauvenet_test(data, point):
    mean = data.mean()           # Mean of incoming array y
    stdv = data.std()            # Its standard deviation
    N = len(data)                # Lenght of incoming arrays
    criterion = 1.0/(2*N)        # Chauvenet's criterion
    d = abs(point-mean)/stdv     # Distance of a value to mean in stdv's
    d /= 2.0**0.5                # The left and right tail threshold values
    prob = erfc(d)               # Area normal dist.
    filter = prob >= criterion   # The 'accept' filter array with booleans
    return filter


def thompson_test(data, point, alpha=0.05):
    sd = np.std(data)
    mean = np.mean(data)
    delta = abs(point - mean)

    t_a2 = stats.t.ppf(1-(alpha/2.), len(data)-2)
    tau = (t_a2 * (len(data)-1)) / (math.sqrt(len(data)) * math.sqrt(len(data)-2 + t_a2**2))

    return delta > tau * 2*sd


def interquartile_range_test(data, point):
    median = np.percentile(data, 50)
    data_low = []
    data_high = []
    for d in data:
        if d <= median:
            data_low.append(d)
        else:
            data_high.append(d)
    q1 = np.percentile(data_low, 50)
    q3 = np.percentile(data_high, 50)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    higher_bound = q3 + 1.5 * iqr
    return point < lower_bound or point > higher_bound


def get_cut_distance(data_link, is_outlier, min_k=float('infinity'), min_dist=0, min_size=3):
    diff_list = list()
    for i in range(1, len(data_link)):
        dist_i_1 = data_link[i-1][2]
        dist_i = data_link[i][2]
        diff_dist = abs(dist_i - dist_i_1)

        if i > min_k:
            # print 'A'
            #return data_link[0][2]
            return np.mean([data_link[0][2], data_link[1][2]])  #, data_link[2][2]])

        # print dist_i, diff_list, diff_dist, is_outlier(np.asarray(diff_list), diff_dist)
        if len(diff_list) > min_size and dist_i >= min_dist and is_outlier(np.asarray(diff_list), diff_dist):
            # print 'B'
            return dist_i_1

        diff_list.append(diff_dist)

        if i <= min_size and dist_i >= min_dist:
            # print 'C'
            return dist_i_1

    return 0.0


class Tosca:

    def __init__(self, kmin=2, kmax=10, xmeans_df=euclidean_distances, singlelinkage_df=euclidean_distance,
                 is_outlier=thompson_test, min_dist=0, verbose=0):
        self.kmin = kmin
        self.kmax = kmax
        self.xmeans_df = xmeans_df
        self.singlelinkage_df = singlelinkage_df
        self.is_outlier = is_outlier
        self.min_dist = min_dist
        self.verbose = verbose

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.k_ = -1
        self.data_ = None

    def fit(self, data):

        # 1st clustering step
        # print 'qui', self.kmin, self.kmax
        xmeans = XMeans(kmin=self.kmin, kmax=self.kmax, distance_function=self.xmeans_df, verbose=self.verbose)
        xmeans.fit(data)
        xmeans_k = xmeans.k_
        xmeans_labels = xmeans.labels_
        xmeans_clusters = points_labels_to_clusters(data, xmeans_labels)
        xmeans_centers = get_clusters_centers(xmeans_clusters, euclidean_distance)

        inverse_xmeans_clusters = dict()
        for xm_cid, xm_pl in xmeans_clusters.iteritems():
            for xm_p in xm_pl:
                inverse_xmeans_clusters[(xm_p[0], xm_p[1])] = xm_cid

        # print 'xmeans k', xmeans_k #, len(xmeans_centers), len(xmeans_labels)

        # 2nd clustering step
        data_dist = pdist(xmeans_centers, metric=self.singlelinkage_df)  # computing the distance
        data_link = linkage(data_dist, method='single', metric=self.singlelinkage_df)  # computing the linkage
        # for a in data_link:
        #     print a
        cut_dist = get_cut_distance(data_link, is_outlier=self.is_outlier, min_k=self.kmin, min_dist=self.min_dist)
        self.cut_dist_ = cut_dist
        # self.cut_dist_ = 35.64  # per plot omino roma
        singlelinkage_labels = fcluster(data_link, cut_dist, 'distance')

        singlelinkage_clusters = points_labels_to_clusters(xmeans_centers, singlelinkage_labels)

        # clustering merge step
        tosca_clusters = defaultdict(list)
        for sl_cid, sl_pl in singlelinkage_clusters.iteritems():
            for sl_p in sl_pl:
                xm_cid = inverse_xmeans_clusters[(sl_p[0], sl_p[1])]
                tosca_clusters[sl_cid].extend(xmeans_clusters[xm_cid])
        tosca_centers = get_clusters_centers(tosca_clusters, euclidean_distance)

        data, tosca_labels = clusters_to_points_labels(tosca_clusters)
        tosca_k = len(tosca_centers)

        self.cluster_centers_ = tosca_centers
        self.labels_ = tosca_labels
        self.k_ = tosca_k
        self.data_ = data

        return self


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


def get_central_point(points):
    c_x = np.mean(points[:, 0])
    c_y = np.mean(points[:, 1])
    return [c_x, c_y]


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


def get_clusters_centers_dict(clusters, dist):
    centers = dict()
    for c, pl in clusters.iteritems():
        centers[c] = get_center(pl, dist)

    return centers


def get_clusters_centers(clusters, dist):
    centers = list()
    for c, pl in clusters.iteritems():
        centers.append(get_center(pl, dist))

    return centers


def get_sse(points, center, dist):
    sse = 0
    for p in points:
        sse += dist(p, center)
    return sse


def get_clusters_sse(clusters, centers, dist):
    sse = 0
    for cid, points in clusters.iteritems():
        sse += get_sse(points, centers[cid], dist)
    return sse


def get_clusters_ssb(clusters, centers, center, dist):
    ssb = 0
    for cid, points in clusters.iteritems():
        ssb += (len(points) * dist(centers[cid], center))
    return ssb


def get_min_max(points):
    min_loc = defaultdict(int)
    max_loc = defaultdict(int)

    for p in points:
        min_loc[encode(float(p[1]), float(p[0]), precision=5)] += 1
        max_loc[encode(float(p[1]), float(p[0]), precision=7)] += 1

    return len(min_loc), len(max_loc)

