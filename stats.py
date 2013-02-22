import numpy as np
import scipy.stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
try:
    import rpy2.robjects as robjects
    r = robjects.r
except ImportError:
    # it's all good
    pass
    


def r_adj_pval(a, meth='BH'):
    """Adjust p-values in R using specified method"""
    robjects.globalenv['unadj_p'] = robjects.FloatVector(
        np.asarray(a).flatten())
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
    fix_nan=True):
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
        print "warning empty data in utest, returning p = 1.0"
        U, p, auroc = 0.0, 1.0, 0.5
    else:
        res = r("wilcox.test(x, y, mu=%r, exact=%s)" % (mu, exact))
        U, p = res[0][0], res[2][0]
        auroc = float(U) / (len(x) * len(y))
    
    # Fix p-value
    if fix_nan and np.isnan(p):
        p = 1.0
    
    # debug
    if verbose:
        print behavior
        s_x = str(robjects.globalenv['x'])
        print s_x[:1000] + '...'
        s_y = str(robjects.globalenv['y'])
        print s_y[:1000] + '...'
        print res
    
    return {'U': U, 'p': p, 'auroc': auroc}

def anova(df, fmla, typ=3):
    # Anova/OLS
    lm = ols(fmla, df=df).fit()
    
    # Grab the pvalues (note we use Type III)
    aov = anova_lm(lm, typ=typ)
    pvals = aov["PR(>F)"]
    pvals.index = map(lambda s: 'p_' + s, pvals.index)
    
    # Grab the explainable sum of squares
    ess = aov.drop("Residual").sum_sq
    ess = ess / ess.sum()
    ess.index = map(lambda s: 'ess_' + s, ess.index)
    
    # Grab the fit
    fit = lm.params
    fit.index = map(lambda s: 'fit_' + s, fit.index)   

    return {'lm':lm, 'aov':aov, 'pvals':pvals, 'ess':ess, 'fit':fit}
    