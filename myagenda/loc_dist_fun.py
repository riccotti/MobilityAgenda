import math
import numpy as np

from scipy.sparse import issparse
from sklearn.utils.extmath import row_norms, safe_sparse_dot
from sklearn.utils import check_array


__author__ = 'Riccardo Guidotti'


# Utility Functions
def _return_float_dtype(X, Y):
    """
    1. If dtype of X and Y is float32, then dtype float32 is returned.
    2. Else dtype float is returned.
    """
    if not issparse(X) and not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if Y is None:
        Y_dtype = X.dtype
    elif not issparse(Y) and not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)
        Y_dtype = Y.dtype
    else:
        Y_dtype = Y.dtype

    if X.dtype == Y_dtype == np.float32:
        dtype = np.float32
    else:
        dtype = np.float

    return X, Y, dtype


# Utility Functions
def check_pairwise_arrays(X, Y):
    X, Y, dtype = _return_float_dtype(X, Y)

    if Y is X or Y is None:
        X = Y = check_array(X, accept_sparse='csr', dtype=dtype)
    else:
        X = check_array(X, accept_sparse='csr', dtype=dtype)
        Y = check_array(Y, accept_sparse='csr', dtype=dtype)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices: "
                         "X.shape[1] == %d while Y.shape[1] == %d" % (
                             X.shape[1], Y.shape[1]))

    return X, Y


# Pairwise distances
def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
    X, Y = check_pairwise_arrays(X, Y)

    if Y_norm_squared is not None:
        YY = check_array(Y_norm_squared)
        if YY.shape != (1, Y.shape[0]):
            raise ValueError(
                "Incompatible dimensions for Y and Y_norm_squared")
    else:
        YY = row_norms(Y, squared=True)[np.newaxis, :]

    if X is Y:  # shortcut in the common case euclidean_distances(X, X)
        XX = YY.T
    else:
        XX = row_norms(X, squared=True)[:, np.newaxis]

    distances = safe_sparse_dot(X, Y.T, dense_output=True)
    distances *= -2
    distances += XX
    distances += YY
    np.maximum(distances, 0, out=distances)

    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0

    return distances if squared else np.sqrt(distances, out=distances)


def spherical_distances(X, Y=None, Y_norm_squared=None, squared=False):

    X, Y = check_pairwise_arrays(X, Y)

    distances = list()

    for x in X:
        dist = list()
        for y in Y:
            dist.append(spherical_distance(x, y))
        distances.append(dist)

    distances = np.array(distances)

    if X is Y:
        distances.flat[::distances.shape[0] + 1] = 0.0

    return distances


def euclidean_distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def spherical_distance(a, b):
    lat1 = a[1]
    lon1 = a[0]
    lat2 = b[1]
    lon2 = b[0]
    R = 6371000.0
    rlon1 = lon1 * math.pi / 180.0
    rlon2 = lon2 * math.pi / 180.0
    rlat1 = lat1 * math.pi / 180.0
    rlat2 = lat2 * math.pi / 180.0
    dlon = (rlon1 - rlon2) / 2.0
    dlat = (rlat1 - rlat2) / 2.0
    lat12 = (rlat1 + rlat2) / 2.0
    sindlat = math.sin(dlat)
    sindlon = math.sin(dlon)
    cosdlon = math.cos(dlon)
    coslat12 = math.cos(lat12)
    f = sindlat * sindlat * cosdlon * cosdlon + sindlon * sindlon * coslat12 * coslat12
    f = math.sqrt(f)
    f = math.asin(f) * 2.0 # the angle between the points
    f *= R
    return f


def euclidean_distance_optics(a, b):
    return euclidean_distance(a.object, b.object)


def spherical_distance_optics(a, b):
    return spherical_distance(a.object, b.object)