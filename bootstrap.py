"""Module containing bootstrap methods for estimating differences between
groups. Loosely based on Efron 1983.
"""
from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import object
from past.utils import old_div

import numpy as np
import pandas

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
    p_value = old_div(np.sum(true_abs_deviation <= abs_deviations), float(n_boots))
    
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
    cdf_at_value = old_div(n_more_extreme, float(len(data)))
    p_at_value = 2 * np.min([cdf_at_value, 1 - cdf_at_value])    
    
    # Optionally deal with p = 0
    if floor and (n_more_extreme == 0 or n_more_extreme == len(data)):
        p_at_value = old_div(2, float(len(data)))
        
        if verbose:
            print("warning: exactly zero p-value encountered in " + \
                "pvalue_of_distribution, flooring")
    
    return p_at_value

class DiffBootstrapper(object):
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
        self.CI_1 = np.percentile(self.means1, (2.5, 97.5))
        self.CI_2 = np.percentile(self.means2, (2.5, 97.5))
        
        # Bootstrapped difference between the groups
        self.diffs = self.means2 - self.means1
        self.CI_diff = np.percentile(self.diffs, (2.5, 97.5))
        
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



# Stimulus-equalizing bootstrap functions

def difference_CI_bootstrap_wrapper(data, **boot_kwargs):
    """Given parsed data from single ulabel, return difference CIs.
    
    data : same format as bootstrap_main_effect expects
    
    Will calculate the following statistics:
        means : mean of each condition, across draws
        CIs : confidence intervals on each condition
        mean_difference : mean difference between conditions
        difference_CI : confidence interval on difference between conditions
        p : two-tailed p-value of 'no difference'
    
    Returns:
        dict of those statistics
    """
    # Yields a 1000 x 2 x N_trials matrix:
    # 1000 draws from the original data, under both conditions.
    bh = bootstrap_main_effect(data, meth=keep, **boot_kwargs)

    # Find the distribution of means of each draw, across trials
    # This is 1000 x 2, one for each condition
    # hist(means_of_all_draws) shows the comparison across conditions
    means_of_all_draws = bh.mean(axis=2)

    # Confidence intervals across the draw means for each condition
    condition_CIs = np.array([
        np.percentile(dist, (2.5, 97.5)) for dist in means_of_all_draws.T])

    # Means of each ulabel (centers of the CIs, basically)
    condition_means = means_of_all_draws.mean(axis=0)

    # Now the CI on the *difference between conditions*
    difference_of_conditions = np.diff(means_of_all_draws).flatten()
    difference_CI = np.percentile(difference_of_conditions, (2.5, 97.5)) 

    # p-value of 0. in the difference distribution
    cdf_at_value = old_div(np.sum(difference_of_conditions < 0.), \
        float(len(difference_of_conditions)))
    p_at_value = 2 * np.min([cdf_at_value, 1 - cdf_at_value])
    
    # Should probably floor the p-value at 1/n_boots

    return {'p' : p_at_value, 
        'means' : condition_means, 'CIs': condition_CIs,
        'mean_difference': difference_of_conditions.mean(), 
        'difference_CI' : difference_CI}

def bootstrap_main_effect(data, n_boots=1000, meth=None, min_bucket=5):
    """Given 2xN set of data of unequal sample sizes, bootstrap main effect.

    We will generate a bunch of fake datasets by resampling from data.
    Then we combine across categories. The total number of data points
    will be the same as in the original dataset; however, the resampling
    is such that each category is equally represented.    
    
    data : list of length N, each entry a list of length 2
        Each entry in `data` is a "category".
        Each category consists of two groups.
        The question is: what is the difference between the groups, without
        contaminating by the different size of each category?
    
    n_boots : number of times to randomly draw, should be as high as you
        can stand
    
    meth : what to apply to the drawn samples from each group
        If None, use means_tester
        It can be any function that takes (group0, group1)
        Results of every call are returned

    Returns:
        np.asarray([meth(group0, group1) for group0, group1 in each boot])    
    """    
    if meth is None:
        meth = means_tester
    
    # Convert to standard format
    data = [[np.asarray(d) for d in dd] for dd in data]
    
    # Test
    alld = np.concatenate([np.concatenate([dd for dd in d]) for d in data])
    if len(np.unique(alld)) == 0:
        raise BootstrapError("no data")
    elif len(np.unique(alld)) == 1:
        raise BootstrapError("all data points are identical")
    
    # How many to generate from each group, total
    N_group0 = np.sum([len(category[0]) for category in data])
    N_group1 = np.sum([len(category[1]) for category in data])
    N_categories = len(data)
    
    # Which categories to draw from
    res_l = []
    for n_boot in range(n_boots):
        # Determine the representation of each category
        # Randomly generating so in the limit each category is equally
        # represented. Alternatively we could use fixed, equal representation,
        # but then we need to worry about rounding error when it doesn't
        # divide exactly evenly.
        fakedata_category_label_group0 = np.random.randint(  
            0, N_categories, N_group0)
        fakedata_category_label_group1 = np.random.randint(
            0, N_categories, N_group1)
        
        # Draw the data, separately by each category
        fakedata_by_group = [[], []]
        for category_num in range(N_categories):
            # Group 0
            n_draw = np.sum(fakedata_category_label_group0 == category_num)
            if len(data[category_num][0]) < min_bucket:
                raise BootstrapError("insufficient data in a category")
            idxs = np.random.randint(0, len(data[category_num][0]),
                n_draw)
            fakedata_by_group[0].append(data[category_num][0][idxs])
            
            # Group 1
            n_draw = np.sum(fakedata_category_label_group0 == category_num)
            if len(data[category_num][1]) < min_bucket:
                raise BootstrapError("insufficient data in a category")
            idxs = np.random.randint(0, len(data[category_num][1]),
                n_draw)
            fakedata_by_group[1].append(data[category_num][1][idxs])
        
        # Concatenate across categories
        fakedata_by_group[0] = np.concatenate(fakedata_by_group[0])
        fakedata_by_group[1] = np.concatenate(fakedata_by_group[1])
        
        # Test difference in means
        #res = np.mean(fakedata_by_group[1]) - np.mean(fakedata_by_group[0])
        res = meth(fakedata_by_group[0], fakedata_by_group[1])
        res_l.append(res)
    
    return np.asarray(res_l)

# Lambdas for bootstrap_main_effect
def means_tester(d0, d1):
    return np.mean(d1) - np.mean(d0)

def keep(d0, d1):
    return (d0, d1)


# Utility bootstrap functions
def CI_compare(CI1, CI2):
    """Return +1 if CI1 > CI2, -1 if CI1 < CI2, 0 if overlapping"""
    if CI1[1] < CI2[0]:
        return -1
    elif CI2[1] < CI1[0]:
        return +1
    else:
        return 0

def simple_bootstrap(data, n_boots=1000, min_bucket=20):
    if len(data) < min_bucket:
        raise BootstrapError("too few samples")
    
    res = []
    data = np.asarray(data)
    for boot in range(n_boots):
        idxs = np.random.randint(0, len(data), len(data))
        draw = data[idxs]
        res.append(np.mean(draw))
    res = np.asarray(res)
    CI = np.percentile(res, (2.5, 97.5))
    
    return res, res.mean(), CI

class BootstrapError(BaseException):
    pass

def bootstrap_CIs_on_dataframe(df):
    """Returns CIs on the mean of each row
    
    Here's how to do it with a groupby:
        res = gobj.apply(lambda df: my.bootstrap.bootstrap_CIs_on_dataframe(df.T))     
    
    Returns: DataFrame
        Index will be the same as df.index
        Columns will be ['lo', 'mean', 'hi', 'mpl_lo', 'mpl_hi']
        matplotlib-style error bars can be extracted as:
        np.array([res['mpl_lo'].values, res['mpl_hi'].values])
    """
    # CIs
    CI_l = []
    CI_keys_l = []
    for idx, sub_data in df.iterrows():
        CI = simple_bootstrap(sub_data)[2]
        CI_l.append(CI)
        CI_keys_l.append(idx)

    #~ agg_err = pandas.DataFrame(CI_l, columns=['lo', 'hi'], 
        #~ index=pandas.Index(CI_keys_l, names=df.index.names))
    
    # The index should be the same as df.index
    agg_err = pandas.DataFrame(CI_l, columns=['lo', 'hi'], 
        index=df.index)
    
    # Mean
    agg_err['mean'] = df.mean(1)
    
    # matplotlib error bars
    agg_err['mpl_lo'] = agg_err['mean'] - agg_err['lo']
    agg_err['mpl_hi'] = agg_err['hi'] - agg_err['mean']
    return agg_err
