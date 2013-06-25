import scipy.spatial
import numpy as np
from matplotlib import mlab
import random

def bootstrapped_intercluster_mahalanobis(cluster1, cluster2, n_boots=1000,
    fix_covariances=True):
    """Bootstrap the intercluster distance.
    
    Returns:
        m - The mean distance
        CI - 95% confidence interval on the distance
        distances - an array of the distances measured on each boot
    """
    d_l = []
    
    # Determine the covariance matrices, or recalculate each time
    if fix_covariances:
        icov1 = np.linalg.inv(np.cov(cluster1, rowvar=0))
        icov2 = np.linalg.inv(np.cov(cluster2, rowvar=0))
    else:
        icov1, icov2 = None, None
    
    # Bootstrap
    for n_boot in range(n_boots):
        # Draw
        idxs1 = np.random.randint(0, len(cluster1), len(cluster1))
        idxs2 = np.random.randint(0, len(cluster2), len(cluster2))
        
        # Calculate and store
        d_l.append(intercluster_mahalanobis(
            cluster1[idxs1], cluster2[idxs2], icov1, icov2))
    
    # Statistics
    d_a = np.asarray(d_l)
    m = np.mean(d_a)
    CI = mlab.prctile(d_a, (2.5, 97.5))
    return m, CI, d_a

def permute_mahalanobis(full_cluster, subcluster_size, n_perms=1000, 
    fix_icov=True, force_idxs=None, seed=0, **mkwargs):
    """Estimate distribution of subcluster distance assuming no effect.
    
    The purpose of this method is to provide statistical context for the
    distance of a subcluster from the full cluster as measured by
    intercluster_mahalanobis. Under the null hypothesis, the subcluster is
    part of the full cluster, and we can find the distribution of expected
    distances using a permutation test. If the true distance is more extreme
    than most of this distribution, then it is likely that the subcluster is
    not part of the full cluster.
    
    full_cluster : 2d array of shape (N_points, N_dimensions)
        Each row is a data point in the full cluster. This should include
        the original cluster as well as the putative subcluster.
    subcluster_size : integer
        Number of points in the subcluster
    n_perms : integer, how many permutations to take
    fix_icov : boolean
        If True then calculate the covariance matrix of the full cluster
        and always use this in calculating the Mahalanobis distance. 
        If False, recalculate the covariance matrix for each permutation.
    seed : if not None, set seed to this for repeatability
    other kwargs : Parameters passed to intercluster_mahalanobis. 
        Mainly you want to make sure that this matches your original call 
        in order to make the permutation test valid.
    
    
    Note: This method is optimized for relatively small subclusters compared
    to the size of the full cluster.
    """
    # Arrayification
    full_cluster = np.asarray(full_cluster)
    
    # Deal with covariance matrix
    if fix_icov:
        icov = np.linalg.inv(np.cov(full_cluster, rowvar=0))
    else:
        icov = None
    full_cluster_size = len(full_cluster)
    
    # Set seed
    if seed:
        random.seed(seed)

    # Permute
    dist_l = []
    for n_perm in range(n_perms):
        # Fake the data
        fake_subcluster_mask = np.zeros(full_cluster_size, dtype=np.bool)
        fake_subcluster_mask[random.sample(
            xrange(full_cluster_size), subcluster_size)] = 1
        fake_sub = full_cluster[fake_subcluster_mask]
        fake_full = full_cluster[~fake_subcluster_mask]
        
        # debug
        if force_idxs is not None:
            fake_sub = full_cluster[force_idxs]
            fake_full = full_cluster[~force_idxs]
        
        # too slow:
        #permed = np.random.permutation(concat_data)
        #fake_sub, fake_full = permed[:len(subcluster)], permed[len(subcluster):]
        
        # Test and store
        dist_l.append(intercluster_mahalanobis(fake_full, fake_sub, icov1=icov,
            **mkwargs))
    dist_a = np.array(dist_l)
    
    return dist_a

def permute_mahalanobis2(full_cluster, subcluster_size1, subcluster_size2, 
    n_perms=1000, fix_icov=True):
    """Estimate distribution of distances between two subclusters.
    
    This is just like permute_mahalanobis but it calculates the distribution
    of distances between two randomly chosen subclusters (of the same
    size as the true subclusters), instead of between the subcluster
    and the full cluster.
    
    Note this hardwires directed=False, and uses intercluster_mahalanobis
    default for use_cluster_center (check this is what you want)
    """
    # Deal with covariance matrix
    if fix_icov:
        icov = np.linalg.inv(np.cov(full_cluster, rowvar=0))
    else:
        icov = None
    full_cluster_size = len(full_cluster)

    # Permute
    dist_l = []
    for n_perm in range(n_perms):
        # Fake the data
        idxs = random.sample(xrange(full_cluster_size),
            subcluster_size1 + subcluster_size2)
        fake_sub1 = full_cluster[idxs[:subcluster_size1]]
        fake_sub2 = full_cluster[idxs[subcluster_size1:]]
        
        # Test and store
        dist_l.append(intercluster_mahalanobis(fake_sub1, fake_sub2, icov1=icov,
            icov2=icov, directed=False))
    dist_a = np.array(dist_l)
    
    return dist_a

def intercluster_mahalanobis(cluster1, cluster2, icov1=None, icov2=None,
    directed=False, use_cluster_center=True, impl='mine', collapse=np.mean):
    """Returns the Mahalanobis distance between the cluster centers.
    
    The Mahalanobis distance between two points, given a specified covariance
    matrix, is calculated by scipy.spatial.distance.mahalanobis. 
    
    In this case the two points are the cluster means. However, the 
    clusters may have different covariances. So we calculate the
    Mahalanobis distance twice, once with each cluster's covariance matrix,
    and return the geometric mean. See `directed`
    
    Arguments:
        cluster1 - 2d array of shape (N_points1, N_dimensions)
        cluster2 - 2d array of shape (N_points2, N_dimensions)
        icov1, icov2 - inverse of the covariance matrix to use for each cluster
            If None, then it will be calculated from the data
            icov2 is ignored if `directed` is True
        directed - boolean
            If True, then use cluster1 as the reference. This means that
            only icov1 is used and that the points in cluster2 are measured
            with respect to the center of cluster1.
            If False, then recursively calls itself twice with directed=True,
            using each subcluster as reference once, then returns the
            harmonic mean.
        use_cluster_center - boolean
            If True, then calculates the distance between cluster2 center
            and cluster1 center. This makes it smaller.
            If False, then calculates the mean distance beween each point
            in cluster2 and the cluster1 center. This makes it more robust (?).
        impl - I reimplemented scipy.spatial.distance.mahalanobis to be
            faster for multidimensional input. Results appear the same
            but much faster.
    
    Returns:
        normalized scalar distance between clusters
    
    Notes
    1.  This measure will be pretty noisy if the size of the reference cluster
        is too small, because the covariance estimate will be noisy.
    2.  This measure is biased upward. Even points drawn from the same 
        distribution will almost certainly have a positive distance.
    
    Example:
    true_cov = np.diag([1, 10]) # elongated along second dimension
    clu1 = np.random.multivariate_normal([0, 0], true_cov, 1000)
    clu2 = np.random.multivariate_normal([1, 0], true_cov, 1005)
    clu3 = np.random.multivariate_normal([0, 1], true_cov, 1010)
    print intercluster_mahalanobis(clu1, clu2)
    print intercluster_mahalanobis(clu1, clu3)
    
    This prints something like:
    1.02827491455
    0.317589712324
    
    clu2 and clu3 are equally far from clu1 in Euclidean space. But
    clu2 is further from clu1 with this metric because the two clusters 
    are separated along the first dimension, in which the clusters are 
    much skinnier.
    """
    # Arrayification
    cluster1 = np.asarray(cluster1)
    cluster2 = np.asarray(cluster2)
    if icov1 is not None:
        icov1 = np.asarray(icov1)
    if icov2 is not None:
        icov2 = np.asarray(icov2)
    
    if not directed:
        # Do it both direction and take harmonic mean
        d1 = intercluster_mahalanobis(cluster1, cluster2, icov1, icov2,
            directed=True, use_cluster_center=use_cluster_center, impl=impl)
        d2 = intercluster_mahalanobis(cluster2, cluster1, icov2, icov1,
            directed=True, use_cluster_center=use_cluster_center, impl=impl)
        return np.sqrt(d1 * d2)
    else:
        # Directed, so just use icov1
        if icov1 is None:
            if len(cluster1) < 2:
                raise ValueError("insufficient data to compute covariance")
            icov1 = np.linalg.inv(np.cov(cluster1, rowvar=0))

        # Which distance to compute
        if use_cluster_center:
            # Distance of cluster 2 center to cluster 1 center, w.r.t. icov1
            d = scipy.spatial.distance.mahalanobis(
                cluster1.mean(axis=0), cluster2.mean(axis=0), icov1)
        else:
            if impl == 'scipy':
                # Avg distance of cluster 2 points to cluster 1 center, w.r.t. icov1
                d_a = np.asarray([scipy.spatial.distance.mahalanobis(
                    cluster1.mean(axis=0), p2, icov1) for p2 in cluster2])
                d = d_a.mean()
            elif impl == 'mine':
                # More efficient way to the same end
                delta = cluster2 - cluster1.mean(axis=0)[None, :]
                d_a = np.sqrt(np.sum(np.dot(delta, icov1) * delta, axis=1))
                if collapse is not None:
                    d = collapse(d_a)
                else:
                    d = d_a
            else:
                raise ValueError('cannot interpret impl: %r' % impl)
        
        return d