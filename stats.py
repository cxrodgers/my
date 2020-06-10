from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
import numpy as np
import scipy.stats
import pandas
import statsmodels.stats.multitest

try:
    import rpy2.robjects as robjects
    r = robjects.r
except ImportError:
    # it's all good
    pass

def adjust_pval(ser, method='fdr_bh'):
    arr = statsmodels.stats.multitest.multipletests(
        ser.values, method=method)[1]
    res = pandas.Series(arr, index=ser.index)
    return res
    
def z2p(zvals):
    """Return the two-tailed p-value corresponding to a z-stat"""
    norm_rv = scipy.stats.norm()
    res = 2 * (1 - norm_rv.cdf(np.abs(zvals)))
    
    try:
        # If zvals is dataframe
        return pandas.DataFrame(res, index=zvals.index, columns=zvals.columns)
    except AttributeError:
        try:
            # If zvals is Series
            return pandas.Series(res, index=zvals.index)
        except AttributeError:
            # If zvals is anything else, like a float
            return res

def pvalue_to_significance_string(pvalue):
    """Return significance string like '*' based on pvalue"""
    if pvalue < .001:
        return '***'
    elif pvalue < .01:
        return '**'
    elif pvalue < .05:
        return '*'
    else:
        return 'n.s.'

def mad_1d(arr1d):
    """My implementation of MAD.
    
    I think there is a bug in statsmodels.robust.mad because the
    median is not subtracted from the data
    """
    deviations = arr1d - np.median(arr1d)
    return old_div(np.median(np.abs(deviations)), .6745)
    
def binom_confint(x=None, n=None, data=None, alpha=.95, meth='beta'):
    """Calculates the confidence interval for binomial data.
    
    Previously I used R's binom.confint with the 'exact' method.
    statsmodels' proportion_confint with the 'beta' method seems to be the
    same.
    """
    if data is not None:
        x = np.sum(data)
        n = len(data)
    
    if meth == 'R_exact':
        r("library(binom)")
        res = r("binom.confint(%d, %d, %f, methods='%s')" % (
            x, n, alpha, 'exact'))
        return res[res.names.index('lower')][0], res[res.names.index('upper')][0]
    else:
        import statsmodels.stats.proportion
        
        # Call statsmodels
        res = list(statsmodels.stats.proportion.proportion_confint(
            x, n, method=meth))
        
        # Bug where nan is sometimes returned
        # https://github.com/statsmodels/statsmodels/pull/2789
        if x == n:
            res[1] = 1
        if x == 0:
            res[0] = 0
        
        return tuple(res)
        

def bootstrap_regress(x, y, n_boot=1000):
    from matplotlib import mlab
    x = np.asarray(x)
    y = np.asarray(y)
    
    m_l, b_l = [], []
    for n in range(n_boot):
        msk = np.random.randint(0, len(x), size=len(x))
        m, b, rval, pval, stderr = scipy.stats.stats.linregress(x[msk], y[msk])
        m_l.append(m)
        b_l.append(b)
    
    res = {
        'slope_m': np.mean(m_l),
        'slope_l': mlab.prctile(m_l, p=2.5),
        'slope_h': mlab.prctile(m_l, p=97.5),
        'intercept_m': np.mean(b_l),
        'intercept_l': mlab.prctile(b_l, p=2.5),
        'intercept_h': mlab.prctile(b_l, p=97.5),
        }
    return res    
    


def r_adj_pval(a, meth='BH'):
    """Adjust p-values in R using specified method"""
    a = np.asarray(a)
    robjects.globalenv['unadj_p'] = robjects.FloatVector(a.flatten())
    return np.array(r("p.adjust(unadj_p, '%s')" % meth)).reshape(a.shape)

def check_float_conversion(a1, a2, tol):
    """Checks that conversion to R maintained uniqueness of arrays.
    
    a1 : array of unique values, typically originating in Python
    a2 : array of unique values, typically grabbed from R
    
    If the lengths are different, or if either contains values that
    are closer than `tol`, an error is raised.
    """
    if len(a1) != len(a2):
        raise ValueError("uniqueness violated in conversion")
    if len(a1) > 1:
        if np.min(np.diff(np.sort(a1))) < tol:
            raise ValueError("floats separated by less than tol")
        if np.min(np.diff(np.sort(a2))) < tol:
            raise ValueError("floats separated by less than tol")

def r_utest(x, y, mu=0, verbose=False, tol=1e-6, exact='FALSE', 
    fix_nan=True, fix_float=False, paired='FALSE'):
    """Mann-Whitney U-test in R
    
    This is a test on the median of the distribution of sample in x minus
    sample in y. It uses the R implementation to avoid some bugs and gotchas
    in scipy.stats.mannwhitneyu.
    
    Some care is taken when converting floats to ensure that uniqueness of
    the datapoints is conserved, which should maintain the ranking.
    
    x : dataset 1
    y : dataset 2
        If either x or y is empty, prints a warning and returns some
        values that indicate no significant difference. But note that
        the test is really not appropriate in this case.
    mu : null hypothesis on median of sample in x minus sample in y
    verbose : print a bunch of output from R
    tol : if any datapoints are closer than this, raise an error, on the
        assumption that they are only that close due to numerical
        instability
    exact : see R doc
        Defaults to FALSE since if the data contain ties and exact is TRUE,
        R will print a warning and approximate anyway
    fix_nan : if p-value is nan due to all values being equal, then
        set p-value to 1.0. But note that the test is really not appropriate
        in this case.
    fix_float : int, or False
        if False or if the data is integer, does nothing
        if int and the data is float, then the data are multiplied by this
        and then rounded to integers. The purpose is to prevent the numerical
        errors that are tested for in this function. Note that differences
        less than 1/fix_float will be removed.
    
    Returns: dict with keys ['U', 'p', 'auroc']
        U : U-statistic. 
            Large U means that x > y, small U means that y < x
            Compare scipy.stats.mannwhitneyu which always returns minimum U
        p : two-sided p-value
        auroc : area under the ROC curve, calculated as U/(n1*n2)
            Values greater than 0.5 indicate x > y
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # What type of R object to create
    if x.dtype.kind in 'iu' and y.dtype.kind in 'iu':        
        behavior = 'integer'
    elif x.dtype.kind == 'f' or y.dtype.kind == 'f':
        behavior = 'float'
    else:
        raise ValueError("cannot determine datatype of x and y")
    
    # Optionally fix float
    if fix_float is True:
        fix_float = 1e6
    if fix_float and behavior == 'float':
        x = np.rint(x * fix_float).astype(np.int)
        y = np.rint(y * fix_float).astype(np.int)
        behavior = 'integer'
    
    # Define variables
    if behavior == 'integer':
        robjects.globalenv['x'] = robjects.IntVector(x)
        robjects.globalenv['y'] = robjects.IntVector(y)
    elif behavior == 'float':
        robjects.globalenv['x'] = robjects.FloatVector(x)
        robjects.globalenv['y'] = robjects.FloatVector(y)
    
        # Check that uniqueness is maintained
        ux_r, ux_p = r("unique(x)"), np.unique(x)
        check_float_conversion(ux_r, ux_p, tol)
        uy_r, uy_p = r("unique(y)"), np.unique(y)
        check_float_conversion(uy_r, uy_p, tol)
        
        # and of the concatenated
        uxy_r, uxy_p = r("unique(c(x,y))"), np.unique(np.concatenate([x,y]))
        check_float_conversion(uxy_r, uxy_p, tol)
    
    # Run the test
    if len(x) == 0 or len(y) == 0:
        print("warning empty data in utest, returning p = 1.0")
        U, p, auroc = 0.0, 1.0, 0.5
    else:
        res = r("wilcox.test(x, y, mu=%r, exact=%s, paired=%s)" % (mu, exact, paired))
        U, p = res[0][0], res[2][0]
        auroc = old_div(float(U), (len(x) * len(y)))
    
    # Fix p-value
    if fix_nan and np.isnan(p):
        p = 1.0
    
    # debug
    if verbose:
        print(behavior)
        s_x = str(robjects.globalenv['x'])
        print(s_x[:1000] + '...')
        s_y = str(robjects.globalenv['y'])
        print(s_y[:1000] + '...')
        print(res)
    
    return {'U': U, 'p': p, 'auroc': auroc}

def anova(df, fmla, typ=3):
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    
    # Anova/OLS
    lm = ols(fmla, df).fit() # 'data' <==> 'df' keyword change with version
    
    # Grab the pvalues (note we use Type III)
    aov = anova_lm(lm, typ=typ)
    pvals = aov["PR(>F)"]
    pvals.index = ['p_' + s for s in pvals.index]
    
    # Grab the explainable sum of squares
    ess = aov.drop("Residual").sum_sq
    ess = old_div(ess, ess.sum())
    ess.index = ['ess_' + s for s in ess.index]
    
    # Grab the fit
    fit = lm.params
    fit.index = ['fit_' + s for s in fit.index]   

    # I think this happens with pathological inputs
    if np.any(aov['sum_sq'] < 0):
        old_div(1,0)

    return {'lm':lm, 'aov':aov, 'pvals':pvals, 'ess':ess, 'fit':fit}
    