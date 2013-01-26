"""Module containing bootstrap methods for estimating differences between
groups. Loosely based on Efron 1983.
"""

import numpy as np
import matplotlib.mlab as mlab

def bootstrap_rms_distance(full_distribution, subset, n_boots=1000, seed=0):
    """Test whether subset of points comes from the larger set
    
    full_distribution : array of shape (n_points, dimensionality)
        The full set of vectors
    subset : array of shape (len_subset, dimensionality)
        A subset of the vectors
    
    The null hypothesis is that the subset comes from the full distribution.
    We take the mean Euclidean distance between each point in the subset 
    and the center of the full distribution as the test statistic.
    
    We then draw `n_boots` fake subsets with replacement from the full
    distribution. The same test statistic is calculated for each fake
    subset. A p-value is obtained by the fraction of draws that are more
    extreme than the true data.
    
    Returns: p_value, subset_distances, bootstrapped_distance_distribution
    """
    np.random.seed(seed)
    
    # true mean of the distribution
    distribution_mean = np.mean(full_distribution, axis=0)
    
    # Draw from the full distribution
    # Each draw contains the same number of samples as the dataset
    # There are `n_boots` draws total (one per row)
    idxs_by_boot = np.random.randint(0, len(full_distribution), 
        (n_boots, len(subset)))
    
    # Actual drawing
    # This will have shape (n_boots, len_dataset, dimensionality)
    draws_by_boot = np.array([
        full_distribution[row_idxs] for row_idxs in idxs_by_boot])
    
    # RMS distance of each row (dim2) from the average
    # This will have shape (n_boots, len_dataset)
    distances_by_boot = np.sqrt(np.mean(
        (draws_by_boot - [[distribution_mean]])**2, axis=2))
    true_distances = np.sqrt(np.mean(
        (subset - [distribution_mean])**2, axis=1))
    
    # Mean RMS distance of each boot (shape n_boots)
    mdistances_by_boot = np.mean(distances_by_boot, axis=1)
    
    # Mean RMS distance of the true subset
    true_mdistance = np.mean(true_distances)
    
    # Now we just take the z-score of the mean distance of the real dataset
    abs_deviations = np.abs(mdistances_by_boot - mdistances_by_boot.mean())
    true_abs_deviation = np.abs(true_mdistance - mdistances_by_boot.mean())
    p_value = np.sum(true_abs_deviation <= abs_deviations) / float(n_boots)
    
    return p_value, true_distances, mdistances_by_boot
    

def pvalue_of_distribution(data, compare=0, floor=True, verbose=True):
    """Returns the two-tailed p-value of `compare` in `data`.
    
    First we choose the more extreme direction: the minimum of 
    (proportion of data points less than compare, 
    proportion of data points greater than compare).
    Then we double this proportion to make it two-tailed.
    
    floor : if True and the p-value is 0, use 2 / len(data)
        This is to account for the fact that outcomes of probability
        less than 1/len(data) will probably not occur in the sample.
    verbose : if the p-value is floored, print a warning
    
    Not totally sure about this, first of all there is some selection
    bias by choosing the more extreme comparison. Secondly, this seems to
    be the pvalue of obtaining 0 from `data`, but what we really want is the
    pvalue of obtaining `data` if the true value is zero.
    
    Probably better to obtain a p-value from permutation test or some other
    test on the underlying data.
    """
    n_more_extreme = np.sum(data < compare)
    cdf_at_value = n_more_extreme / float(len(data))
    p_at_value = 2 * np.min([cdf_at_value, 1 - cdf_at_value])    
    
    # Optionally deal with p = 0
    if floor and (n_more_extreme == 0 or n_more_extreme == len(data)):
        p_at_value = 2 / float(len(data))
        
        if verbose:
            print "warning: exactly zero p-value encountered in " + \
                "pvalue_of_distribution, flooring"
    
    return p_at_value

class DiffBootstrapper:
    """Object to estimate the difference between two groups with bootstrapping."""
    def __init__(self, data1, data2, n_boots=1000, min_bucket=5):
        self.data1 = data1
        self.data2 = data2
        self.n_boots = n_boots
        self.min_bucket = min_bucket
    
    def execute(self, seed=0):
        """Test the difference in means with bootstrapping.
        
        Data is drawn randomly from group1 and group2, with resampling.
        From these bootstraps, estimates with confidence intervals are 
        calculated for the mean of each group and the difference in means.
        
        The estimated difference is positive if group2 > group1.
        
        Sets: mean1, CI_1, mean2, CI_2, diff_estimate, diff_CI, p1, p2
        
        p1 is the p-value estimated from the distribution of differences
        p2 is the p-value from a 1-sample ttest on that distribution
        """
        if len(self.data1) < self.min_bucket or len(self.data2) < self.min_bucket:
            #~ raise BootstrapError(
                #~ 'insufficient data in bucket in bootstrap_two_groups')
            raise ValueError(
                'insufficient data in bucket in bootstrap_two_groups')
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random samples, shape (n_boots, len(group))
        self.idxs1 = np.random.randint(0, len(self.data1), 
            (self.n_boots, len(self.data1)))
        self.idxs2 = np.random.randint(0, len(self.data2), 
            (self.n_boots, len(self.data2)))
        
        # Draw from the data
        self.draws1 = self.data1[self.idxs1]
        self.draws2 = self.data2[self.idxs2]
        
        # Bootstrapped means of each group
        self.means1 = self.draws1.mean(axis=1)
        self.means2 = self.draws2.mean(axis=1)
        
        # CIs on group means
        self.CI_1 = mlab.prctile(self.means1, (2.5, 97.5))
        self.CI_2 = mlab.prctile(self.means2, (2.5, 97.5))
        
        # Bootstrapped difference between the groups
        self.diffs = self.means2 - self.means1
        self.CI_diff = mlab.prctile(self.diffs, (2.5, 97.5))
        
        # p-value
        self.p_from_dist = pvalue_of_distribution(self.diffs, 0)
        
        # save memory
        del self.idxs1
        del self.idxs2
        del self.draws1
        del self.draws2
    
    @property
    def summary_group1(self):
        """Return mean, CI_low, CI_high on group1"""
        return self.means1.mean(), self.CI_1[0], self.CI_1[1]
    
    @property
    def summary_group2(self):
        return self.means2.mean(), self.CI_2[0], self.CI_2[1]
    
    @property
    def summary_diff(self):
        return self.diffs.mean(), self.CI_diff[0], self.CI_diff[1]
    
    @property
    def summary(self):
        return list(self.summary_group1) + list(self.summary_group2) + \
            list(self.summary_diff) + [self.p_from_dist]