from myagenda.loc_dist_fun import *
from myagenda.kmeans import *

__author__ = 'Riccardo Guidotti'


def bic(clusters, centroids):
    num_points = sum(len(cluster) for cluster in clusters)
    num_dims = clusters[0][0].shape[0]

    log_likelihood = _loglikelihood(num_points, num_dims, clusters, centroids)
    num_params = _free_params(len(clusters), num_dims)

    return log_likelihood - num_params / 2.0 * np.log(num_points)


def _free_params(num_clusters, num_dims):
    return num_clusters * (num_dims + 1)


def _loglikelihood(num_points, num_dims, clusters, centroids):
    ll = 0
    for cluster in clusters:
        fRn = len(cluster)
        t1 = fRn * np.log(fRn)
        t2 = fRn * np.log(num_points)
        variance = _cluster_variance(num_points, clusters, centroids) or np.nextafter(0, 1)
        t3 = ((fRn * num_dims) / 2.0) * np.log((2.0 * np.pi) * variance)
        t4 = (fRn - 1.0) / 2.0
        ll += t1 - t2 - t3 - t4
    return ll


def _cluster_variance(num_points, clusters, centroids):
    s = 0
    denom = float(num_points - len(centroids))
    for cluster, centroid in zip(clusters, centroids):
        distances = euclidean_distances(cluster, centroid)
        s += (distances*distances).sum()
    return s / denom if 0 != denom else 0


class XMeans:

    def __init__(self, kmin=2, kmax=10, distance_function=euclidean_distances,
                 n_init=10, cluster_centers='k-means++', max_iter=300,
                 tol=1e-4, precompute_distances=True,
                 verbose=0, random_state=None, copy_x=True, n_jobs=1):
        self.kmin = kmin
        self.kmax = kmax
        self.distance_function = distance_function
        self.original_n_init = n_init
        self.n_init = n_init
        self.cluster_centers = cluster_centers
        self.max_iter = max_iter
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.labels = None

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.k_ = -1

    def fit(self, data):
        k = self.kmin
        best_k = -1
        best_bic = - float('inf')
        best_inertia = -float('inf')
        best_centers = None
        while k <= self.kmax:
            if self.verbose:
                print 'Fitting with k=%d' % k

            k_means = KMeans(init=self.cluster_centers, n_clusters=k, n_init=self.n_init,
                             max_iter=self.max_iter, tol=self.tol, precompute_distances=self.precompute_distances,
                             verbose=False, random_state=self.random_state, copy_x=self.copy_x,
                             n_jobs=self.n_jobs)
            k_means.fit(data, distance_function=self.distance_function)
            self.labels = k_means.labels_
            self.cluster_centers = k_means.cluster_centers_

            centroid_distances = self.distance_function(self.cluster_centers, self.cluster_centers)
            centroid_distances += np.diag([np.Infinity] * k)
            centroids_range = centroid_distances.min(axis=-1)

            new_cluster_centers = []
            for i, centroid in enumerate(self.cluster_centers):
                if self.verbose:
                    print '\tSplitting cluster %d / %d (k=%d)' % ((i+1), len(self.cluster_centers), k)
                # literature version
                direction = np.random.random(centroid.shape)
                vector = direction * (centroids_range[i] / np.sqrt(direction.dot(direction)))
                new_point1 = centroid + vector
                new_point2 = centroid - vector

                if self.verbose:
                    print '\tRunning secondary kmeans'

                model_index = (self.labels == i)

                if not np.any(model_index):
                    if self.verbose:
                        print '\tDisregarding cluster since it has no citizens'
                    continue

                points = data[model_index]

                if len(points) == 1:
                    if self.verbose:
                        print '\tCluster made by only one citizen'
                    new_cluster_centers.append(centroid)
                    continue

                child_k_means = KMeans(init=np.asarray([new_point1, new_point2]), n_clusters=2, n_init=1)
                # child_k_means = KMeans(init='k-means++', n_clusters=2, n_init=10,
                #              max_iter=self.max_iter, tol=self.tol, precompute_distances=self.precompute_distances,
                #              verbose=False, random_state=self.random_state, copy_x=self.copy_x,
                #              n_jobs=self.n_jobs)
                child_k_means.fit(points, distance_function=self.distance_function)
                cluster1 = points[child_k_means.labels_ == 0]
                cluster2 = points[child_k_means.labels_ == 1]

                if len(cluster1) == 0 or len(cluster2) == 0:
                    if self.verbose:
                        print '\tUsing parent'
                    new_cluster_centers.append(centroid)
                    continue

                bic_parent = bic([points], [centroid])
                bic_child = bic([cluster1, cluster2], child_k_means.cluster_centers_)

                if self.verbose:
                    print '\tbic_parent = %f, bic_child = %f' % (bic_parent, bic_child)

                if bic_child > bic_parent:
                    if self.verbose:
                        print '\tUsing children'
                    new_cluster_centers.append(child_k_means.cluster_centers_[0])
                    new_cluster_centers.append(child_k_means.cluster_centers_[1])
                else:
                    if self.verbose:
                        print '\tUsing parent'
                    new_cluster_centers.append(centroid)

            if k == len(new_cluster_centers):
                # self.cluster_centers = np.array(new_cluster_centers)
                # k_means = KMeans(init=self.cluster_centers, n_clusters=k, n_init=1,
                #              max_iter=self.max_iter, tol=self.tol, precompute_distances=self.precompute_distances,
                #              verbose=self.verbose, random_state=self.random_state, copy_x=self.copy_x,
                #              n_jobs=self.n_jobs)
                # k_means.fit(data, distance_function=self.distance_function)
                # inertia_k = k_means.inertia_
                # if inertia_k > best_inertia:
                #     best_inertia = inertia_k
                #     best_centers = self.cluster_centers
                #     best_k = k

                bic_k = bic([data], new_cluster_centers)
                if bic_k > best_bic:
                    best_bic = bic_k
                    best_centers = self.cluster_centers
                    best_k = k

                k += 1
                self.cluster_centers = 'k-means++'
                self.n_init = self.original_n_init
            else:
                k = len(new_cluster_centers)
                self.cluster_centers = np.array(new_cluster_centers)
                self.n_init = 1

        if best_k == -1:
            best_k = self.kmax
            best_centers = 'k-means++'
            self.n_init = self.original_n_init
        else:
            self.n_init = 1

        if self.verbose:
            print 'Refining model with k = %d' % best_k

        k_means_final = KMeans(init=best_centers, n_clusters=best_k, n_init=self.n_init,
                             max_iter=self.max_iter, tol=self.tol, precompute_distances=self.precompute_distances,
                             verbose=self.verbose, random_state=self.random_state, copy_x=self.copy_x,
                             n_jobs=self.n_jobs)
        k_means_final.fit(data, distance_function=self.distance_function)
        self.cluster_centers_ = k_means_final.cluster_centers_
        self.labels_ = k_means_final.labels_
        self.inertia_ = k_means_final.inertia_
        self.k_ = best_k

        return self
