from builtins import zip
from builtins import range
import pandas
import numpy as np
import my.plot 
import matplotlib.pyplot as plt
import os
import sklearn.linear_model

class ConvergenceError(Exception):
    """Raised when decoder fails to converge"""
    pass

def to_indicator_df(ser, bins=None, propagate_nan=True):
    """Bin series and convert to DataFrame of indicators
    
    ser : series of data
    bins : how to bin the data
        If None, assume the ser is already binned (or labeled)
    propagate_nan : bool
        If True, then wherever ser is null, that row will be all null
        in return value
    
    Returns : DataFrame
        Rows corresponding to values in `ser` outside `bins` 
        will be all zero (I think).
    """
    # Cut
    if bins is not None:
        binned = pandas.cut(ser, bins, labels=False)
        unique_bins = binned.dropna().unique().astype(np.int)
    else:
        binned = ser
        unique_bins = binned.dropna().unique()
    
    # Indicate
    indicated_l = []
    for bin in unique_bins:
        indicated = (binned == bin).astype(np.int)
        indicated_l.append(indicated)
    indicator_df = pandas.DataFrame(np.transpose(indicated_l), 
        index=ser.index, columns=unique_bins).sort_index(axis=1)
    
    # Propagate nan
    if propagate_nan and ser.isnull().any():
        indicator_df.loc[ser.isnull(), :] = np.nan
    
    return indicator_df

def indicate_series(series):
    """Different version of to_indicator_df"""
    # Indicate each unique variable within `series`
    indicated_l = []
    indicated_keys_l = []
    
    # Get unique
    unique_variables = sorted(series.unique())

    # Indicate
    for unique_variable in unique_variables:
        indicated_l.append((series == unique_variable).astype(np.int))
        indicated_keys_l.append(unique_variable)

    # DataFrame it
    df = pandas.concat(indicated_l, keys=indicated_keys_l, 
        names=[series.name], axis=1, verify_integrity=True)
    
    return df
    

def intify_classes(session_classes, by=('rewside', 'choice')):
    """Return integer class label for each row in `session_classes`.
    
    session_classes : DataFrame with columns rewside, choice, servo_pos
    
    by : tuple or None
        Acceptable values:
        ('shape',)
        ('rewside', 'choice') 
        ('rewside', 'choice', 'servo_pos') 
        None
    
    Returns : Series
        index: same as session_classes.index
        values: int
            if by is None, all values are zero
            Otherwise, it will be the class id, which varies from 
            0-1 if by is ('shape',) or 
            0-3 if by is ('rewside', 'choice') or 
            0-11 if by is ('rewside', 'choice', 'servo_pos')
    """
    # Error check
    assert by in [
        None,
        ('shape',),
        ('rewside',),
        ('choice',),
        ('rewside', 'choice'), 
        ('rewside', 'choice', 'servo_pos'),
        ]
    
    # Initialize to zero
    intified_session_classes = pandas.Series(
        np.zeros(len(session_classes), dtype=np.int),
        index=session_classes.index,
        )
    
    # If by is None we just return zeros
    # Otherwise do this
    if by is not None:
        # Replace each column with integers
        replacing_dict = {
            'shape': {'concave': 0, 'convex': 1},
            'rewside': {'left': 0, 'right': 1}, 
            'choice': {'left': 0, 'right': 1}, 
            'servo_pos': {1670: 0, 1760: 1, 1850: 2}
        }
        coded_session_classes = session_classes.replace(replacing_dict)

        # Add factor * each column
        if 'shape' in by:
            # Also least significant bit
            intified_session_classes += coded_session_classes['shape']
        
        if 'rewside' in by:
            # Least significant bit
            intified_session_classes += coded_session_classes['rewside']

        if 'choice' in by:
            # Middle (or most significant)
            intified_session_classes += 2 * coded_session_classes['choice']
        
        if 'servo_pos' in by:
            # Multiply by 4 because 4 possible values for each servo_pos
            # Slowest-changing bit
            intified_session_classes += 4 * coded_session_classes['servo_pos']

    return intified_session_classes

def normalize_features(session_features):
    """Normalize features
    
    Returns: norm_session_features, normalizing_mu, normalizing_sigma
    """
    norm_session_features = session_features.copy()
    
    # Demean
    normalizing_mu = norm_session_features.mean()
    norm_session_features = norm_session_features - normalizing_mu

    # Fill with zero here so that NaNs cannot impact the result
    norm_session_features = norm_session_features.fillna(0)            
    
    # Scale, AFTER fillna, so that the scale is really 1
    normalizing_sigma = norm_session_features.std()
    norm_session_features = norm_session_features / normalizing_sigma
    
    # Fillna again, in the case that every feature was the same
    # in which case it all became inf after scaling
    norm_session_features = norm_session_features.fillna(0)            
    norm_session_features = norm_session_features.replace(
        {-np.inf: 0, np.inf: 0})

    return norm_session_features, normalizing_mu, normalizing_sigma

def stratify_and_calculate_sample_weights(strats):
    """Calculate the weighting of each sample
    
    The samples are weighted in inverse proportion to the 
    frequency of the corresponding value in `strats`.
    
    Returns: strat_id2weight, sample_weights
        strat_id2weight : dict of strat value to weight
        sample_weights : weight of each sample (mean 1)
    """
    # Calculate weight of each strat_id
    strat_id2weight = {}
    for this_strat in np.unique(strats):
        n_this_strat = np.sum(strats == this_strat)
        weight_this_strat = len(strats) / float(n_this_strat)
        strat_id2weight[this_strat] = weight_this_strat
    
    # Sample weights
    sample_weights = np.array([strat_id2weight[strat_id]
        for strat_id in strats])
    
    # Make them mean 1
    sample_weights = sample_weights / sample_weights.mean()
    
    # Return
    return strat_id2weight, sample_weights

def stratified_split_data(stratifications, n_splits=3, 
    shuffle=False, random_seed=None, n_tune_splits=1,
    ):
    """Stratify data into different groups, equalizing by class.
    
    This defines `n_splits` "splits" of the data. Within each split, each
    row is assigned to "train", "test", or (optionally) "tune" sets. Care
    is taken to equalize the representation of each stratification in each
    set of each split, as much as possible.
    
    This is an example of the results applied to a small dataset.
    
    split            0      1      2      3      4      5      6
    strat trial                                                 
    0     57     train  train  train  train   test   tune  train
          80     train  train  train  train  train   test   tune
          90      tune  train  train  train  train  train   test
    3     61     train  train   test   tune  train  train  train
          64     train  train  train   test   tune  train  train
          98     train  train  train  train   test   tune  train
          109    train  train  train  train  train   test   tune
          117     tune  train  train  train  train  train   test

    Because there are only 3 trials in stratification 0, some splits will
    have no representatives of this stratification in the testing set.
    If there are at least `n_splits` examples of each stratification, then
    each split will have at least one example in the test set.
    
    If all of the stratifications are smaller than `n_splits`, then it is
    possible that one of the splits will have no testing or tuning set at all.
    
    Everything is implemented with circular shifts to equalize representation
    as much as possible.
    
    Each row is in the test set on exactly one split, and in the tuning
    set on exactly `n_tune_splits` splits. It is in the training set on
    the remaining splits. `n_tune_splits` can be zero.
    
    Each stratification is handled completely separately from the others.
    To break symmetry, the first example in each stratification is assigned
    to be in the test set on a random split, and from then on the shift
    is circular. Without this randomness, split 0 would tend to have larger
    test sets than the remaining splits.
    
    
    stratifications : Series
        index : however the data is indexed, e.g., session * trial
        values : class of that row
        If no stratification is desired, just provide the same value for each
        pandas.Series(np.ones(len(big_tm)), index=big_tm.index)
    
    n_splits : int
        Number of splits, and number of columns in the result.
    
    n_tune_splits : int
        Number of splits each row should be in the tuning set.
        Must be in the range [0, n_splits - 2].
    
    shuffle : bool
        If True, randomize the order of the trials within each stratification
        before applying the circular shift.
    
    random_seed : int, or None
        If not None, call np.random.seed(random_seed)
        If None, do not change random seed.
    
    group_names : list
        The names of the other datasets
        Names can be repeated in order to vary the relative sizes
        So for instance if n_splits = 5, and group_names is
        ['train', 'train', 'train', 'tune']
        Then there will be 3 parts training, 1 part tuning, and 1 part testing
        The allocation is done by modding the indices, not randomly sampling,
        so the results are always nearly exactly partitioned equally.
    
    
    Returns : DataFrame
        index : same as stratifications.index
        columns : range(n_splits)
        values : 'test', 'tune', or 'train'
    """
    ## Initialize
    # Set state
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Set group_names
    # This determines the relative prevalence of each set
    # And the relationship between adjacent splits
    n_train_splits = n_splits - n_tune_splits - 1
    assert n_train_splits > 0
    assert n_tune_splits >= 0
    group_names = (
        ['test'] + 
        n_tune_splits * ['tune'] + 
        n_train_splits * ['train'])
    
    # Identify the unique values of `stratifications`
    unique_strats = np.sort(np.unique(stratifications))
    
    
    ## Define a random strat_offset for each stratification
    # Choose a value in range(n_splits) for each value in unique_strats
    # Use permutations to avoid repeating too frequently
    # This many runs through range(n_splits)
    n_perms = int(np.ceil(len(unique_strats) / n_splits))
    
    # Concatenate each run
    strat_offsets = np.concatenate([
        np.random.permutation(range(n_splits))
        for n_perm in range(n_perms)])
    
    # Truncate to same length
    strat_offsets = strat_offsets[:len(unique_strats)]
    
    
    ## Generate the group_shift of each entry in stratifications
    # Consider each stratification separately
    group_shift_of_each_index_l = []
    group_shift_of_each_index_keys_l = []
    for strat, strat_offset in zip(unique_strats, strat_offsets):
        # Find the corresponding indices into stratifications
        # Note: raw indices, not values in stratifications.index
        indices = np.where(stratifications == strat)[0]
        
        # Shufle them
        if shuffle:
            np.random.shuffle(indices)
        
        # Assign each value in `indices` a "shift"
        # This is the split on which that row will be in the test set
        # It can also be interpreted as the circular shift to apply
        # to `group_names` to get the set for each split for this row.
        # This works because 'test' is always first in 'group_names'. 
        #
        # Also use strat_offset here to uniformly shift all rows
        # Otherwise the first row in this stratification always gets assigned
        # to the test set on split 0, which is a problem because split 0 will
        # thus always have the largest test set.
        group_shift_of_each_index = np.mod(
            strat_offset + np.arange(len(indices), dtype=np.int), 
            n_splits)
        
        # Convert to Series and store
        group_shift_of_each_index_l.append(
            pandas.Series(group_shift_of_each_index, 
            index=stratifications.iloc[indices].index))
        group_shift_of_each_index_keys_l.append(strat)
    
    # Concat
    # Indexed now by strat * trial; no longer in order of `stratifications`
    group_shift_ser = pandas.concat(
        group_shift_of_each_index_l, 
        keys=group_shift_of_each_index_keys_l, names=['strat'],
        ).sort_index()


    ## Use group_shift to assign each row its groups
    # Columns: splits
    # Index: strat * trial
    # Values: name of set (test, tune, train)
    # Each row is just a circular shift of group_names
    set_by_split_strat_and_trial = pandas.DataFrame(
        [np.roll(group_names, shift) for shift in group_shift_ser.values], 
        index=group_shift_ser.index, 
        columns=pandas.Series(range(n_splits), name='split')
        )
    
    # Debugging: count set sizes
    # For stratifications with fewer examples than `n_splits`,
    # any given split may have no examples in one of the sets.
    # In principle, if all the stratifications are smaller than `n_splits`,
    # it's possible that some split may have no training or tuning examples
    # for all stratifications
    # Check: set_sizes.sum(level='split')
    set_sizes = set_by_split_strat_and_trial.stack().groupby(
        ['strat', 'split']).value_counts().unstack().fillna(0).astype(np.int)
    
    # Drop the 'strat' level and sort by trial to return
    res = set_by_split_strat_and_trial.droplevel('strat').sort_index()
    assert res.index.equals(stratifications.index)

    return res

def logregress2(
    features, labels, train_indices, test_indices,
    sample_weights=None, strats=None, regularization=10**5,
    testing_set_name='test', max_iter=10000, non_convergence_action='error',
    solver='liblinear', tol=1e-6, min_training_points=10,
    ):
    """Run cross-validated logistic regression 
       
    testing_set_name : this is used to set the values in the 'set' column
        of the per_row_df, and also in scores_df
        If this is a tuning set, pass 'tune'
    
    solver, max_iter, tol : passed to sklearn.linear_model.LogisticRegression
        With solver == 'lbfgs', I had to use huge n_iter (1e6) and even
        then sometimes got gradient errors (LNSRCH). Going back to 'liblinear'.
    
    non_convergence_action : string
        if 'error': raise error when doesn't converge
        if 'pass': do nothing
        
        This doesn't catch all ConvergenceWarning, like LNSRCH
        For that, do this:
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.simplefilter("error", ConvergenceWarning)        
    
    Returns: dict
        'weights': logreg.coef_[0],
        'intercept': logreg.intercept_[0],
        'scores_df': scores_df,
        'per_row_df': res_df,       
    """
    ## Error check
    if len(train_indices) < min_training_points:
        raise ValueError(   
            "must provide at least {} training examples, I got {}".format(
            min_training_points, len(train_indices)))

    if len(np.unique(test_indices)) <= 1:
        raise ValueError("must provide more than one label type")
    
    
    ## Split out test and train sets
    X_train = features[train_indices]
    X_test = features[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    
    
    ## Set up sample weights
    ## Note that this doesn't know about missing stratifications, so
    ## if there's a strat missing, it just will be ignored
    # Get them if they don't exist
    if sample_weights is None:
        # Use strat if available, otherwise use 1
        if strats is None:
            sample_weights = np.ones(len(features))
        else:
            strat_id2weight, sample_weights = (
                stratify_and_calculate_sample_weights(strats)
            )
    
    # Split
    sample_weights_train = sample_weights[train_indices]
    sample_weights_test = sample_weights[test_indices]    
    

    ## Fit
    # Initialize fitter
    logreg = sklearn.linear_model.LogisticRegression(
        C=(1.0 / regularization),
        max_iter=max_iter, solver=solver, tol=tol,
    )

    # Fit, applying the sample weights
    logreg.fit(
        X_train, y_train, sample_weight=sample_weights_train, 
        )
    
    # Deal with non-convergence
    if logreg.n_iter_.item() >= logreg.max_iter:
        if non_convergence_action == 'error':
            raise ConvergenceError("non-convergence in decoder")
        elif non_convergence_action == 'pass':
            pass
        else:
            raise ValueError("unknown non_convergence_action: {}".format(
                non_convergence_action))

    
    ## Extract results
    # The per-row predictions and scores
    predictions = logreg.predict(features)
    res_df = pandas.DataFrame.from_dict({
        'dec_function': logreg.decision_function(features),
        'proba': logreg.predict_proba(features)[:, 1],
        'prediction': predictions,
        'pred_correct': (predictions == labels).astype(np.float),
        'actual': labels,
        'sample_weight': sample_weights,
        })
    res_df['weighted_correct'] = res_df['sample_weight'] * res_df['pred_correct']
    
    # Assign train or test set to each row
    res_df['set'] = ''
    res_df['set'].values[test_indices] = testing_set_name
    res_df['set'].values[train_indices] = 'train'
    
    # The overall scores
    scores_df = res_df.loc[
        res_df['set'].isin([testing_set_name, 'train']),
        ['set', 'pred_correct', 'weighted_correct']
        ].groupby('set').mean()
    
    
    ## Return
    output = {
        'weights': logreg.coef_[0],
        'intercept': logreg.intercept_[0],
        'scores_df': scores_df,
        'per_row_df': res_df,
    }
    return output


def tuned_logregress(folds, norm_session_features, intified_labels,
    sample_weights, reg_l):
    """Train over all splits and all regularizations, evaluate on the tuning set

    If some stratifications are very small, they will be missing from the
    test, tune, or train set on some splits. When it's missing from
    test set or tune set, it's not so bad, because in the end we combine
    data from the split where each trial was in the test/tune set, so
    they're all included. It's bad when it's missing from the training set,
    because it won't affect the coefficients, but also because the weights
    are wrong (ill-posed) with a missing stratification. The use of high-fold
    cross-validation helps: in the limit with leave-one-out, it will only
    be missing from the training set once.
    """
    ## Iterate over splits
    tuning_results_l = []
    tuning_scores_l = []
    tuning_keys_l = []
    
    for split in folds.columns:
        ## Extract data masks and indices
        test_data_mask = folds.loc[:, split] == 'test'
        tune_data_mask = folds.loc[:, split] == 'tune'
        train_data_mask = folds.loc[:, split] == 'train'
        
        test_indices = np.where(test_data_mask.values)[0]
        tune_indices = np.where(tune_data_mask.values)[0]
        train_indices = np.where(train_data_mask.values)[0]
        
        
        ## Iterate over regularizations
        for reg in reg_l:
            # Train on the training set, test on the tuning set
            logreg_res = logregress2(
                features=norm_session_features.values, 
                labels=intified_labels.values, 
                train_indices=train_indices, 
                test_indices=tune_indices,
                sample_weights=sample_weights,
                regularization=10 ** reg,
                testing_set_name='tune',
                )                    
            
            # Rename the sets, because  the unused rows marked with 
            # '' were actually 'test'
            logreg_res['per_row_df']['set'] = (
                logreg_res['per_row_df']['set'].replace(
                {'': 'test'})
            )
            
            # Add the index back
            logreg_res['per_row_df'].index = norm_session_features.index
            logreg_res['weights'] = pandas.Series(logreg_res['weights'],
                index=norm_session_features.columns, name='weights')

            # Error check
            assert (logreg_res['per_row_df']['set'] == folds.loc[:, split]).all()
           
            # Store
            tuning_results_l.append(logreg_res)
            tuning_keys_l.append((split, reg))
    
    return tuning_keys_l, tuning_results_l
    
def recalculate_decfun_standard(features, mu, sigma, weights, intercepts):
    """Recalculate the decision function from features in the standard way.
    
    In the standard approach, we first standardize the features (zero mean and
    unit variance), multiply by the weights, and add the intercept.
    
    features : DataFrame of shape (n_trials, n_features)
        index: MultiIndex (session, trial)
        columns: MultiIndex of features
        These are the raw features. They can contain null values.
    
    mu, sigma : DataFrame of shape (n_sessions, n_features)
        index: session
        columns: MultiIndex of features
        These are the mean and standard deviation of the raw features.
        When there is no data in features, mu will be null, and standard
        deviation should be zero.
        filled with zeros.
    
    weights : DataFrame of shape (n_sessions * n_decode_labels, n_features)
        index : MultiIndex (session, decode_label)
        columns : MultiIndex of features
        These are the coefficients from the model. They should not be null.
    
    intercepts : Series of length (n_sessions * n_decode_labels)
        index : MultiIndex (session, decode_label)
    
    Returns: Series
        index : MultiIndex (session, decode_label, trial)
        This is the decision function for each trial. 
    """
    # Debugging: this was for a single session * label
    #~ srecalc_decfun = sfeatures.sub(snmu).divide(snsig).mul(sweights).fillna(
        #~ 0).sum(1) + sicpt

    return features.sub(mu).divide(sigma).mul(weights).fillna(
        0).sum(1).add(intercepts)

def recalculate_decfun_raw(features, mu, sigma, weights, intercepts):
    """Recalculate the decision function from features in the raw way.
    
    This is a reordering of the standard formula, so that the weights
    are directly interpretable as the effect of a single instance of
    a feature (e.g., a single contact).
    
    Because we want to use the raw non-standardized features, we have to
    apply the feature scaling to the weights. These "scaled weights" are
    the weights divided by the standard deviation of the features. We
    also have to account for this change in the intercept.
    
    decfun = weights * (features - mu) / sigma + intercept
    decfun = (weights / sigma) * features + (-mu * weights / sigma) + intercept
    decfun = scaled_weights * features + (-scaled_weights * mu + intercept)
    decfun = scaled_weights * features + icpt_transformed
    
    The inputs are the same as recalculate_decfun_standard.
    
    Returns: scaled_weights, icpt_transformed, decfun
        scaled_weights : DataFrame, same shape as weights
            The scaled weights
        icpt_transformed : Series, same shape as intercepts
            The transformed intercepts
        decfun : Series, same shape as the result of recalculate_decfun_standard
            The recalculated decision function
    """
    # Debugging: this was for a single session * label
    #~ sweights_unscaled = sweights.divide(snsig).fillna(0)
    #~ sicpt_transformed = -sweights_unscaled.mul(snmu).fillna(0).sum() + sicpt
    #~ srecalc_decfun2 = (
        #~ sfeatures.fillna(sfeatures.mean()).mul(sweights_unscaled).fillna(0).sum(1) 
        #~ + sicpt_transformed
    #~ )
    
    # Scale the weights
    scaled_weights = weights.divide(sigma).fillna(0)

    # Transform the intercepts
    icpt_transformed = (-scaled_weights.mul(mu).fillna(0).sum(1)).add(
        intercepts)

    # Must pre-fill model_features with the session mean of each feature
    filled_features = pandas.concat([
        features.loc[session].fillna(features.loc[session].mean())
        for session in features.index.levels[0]],
        axis=0, keys=features.index.levels[0]
    )

    # Recalculate
    decfun = filled_features.mul(scaled_weights).fillna(0).sum(1).add(
        icpt_transformed)
    
    return scaled_weights, icpt_transformed, decfun

def recalculate_decfun_partitioned(features, mu, sigma, weights, intercepts,
    raw_mask):
    """Recalculate the decision function using some raw and some scaled.
    
    This is a combination of the approaches in recalculate_decfun_standard
    and recalculate_decfun_raw. Some "raw" features are handled in the raw
    way and the rest are handled in the scaled way.
    
    The inputs are the same as recalculate_decfun_standard, except for:
    
    raw_mask : boolean Series, same shape as the feature columns
        True for raw features
    
    Returns: features_part, weights_part, icpt_transformed, decfun
        features_part : DataFrame, same shape as features
            The transformed features. Raw features are unchanged from `features`,
            and the rest are scaled as in the standard way.
        weights_part : DataFrame, same shape as weights
            The transformed weights. The weights of raw features are scaled,
            and the rest are the the same as in `weights`.
        icpt_transformed : Series, same shape as intercepts
            The transformed intercepts, now including the effect of the
            raw weights.
        decfun : Series, same shape as the result of recalculate_decfun_standard
            The recalculated decision function
    """
    # Debugging
    #~ sraw_mask = sweights.index.get_level_values('family') == 'summed_count'
    #~ sfeatures_part = sfeatures.copy()
    #~ sweights_part = sweights.copy()
    #~ sfeatures_part = sfeatures_part.fillna(sfeatures_part.mean())
    #~ sfeatures_part.loc[:, ~sraw_mask] = sfeatures_part.loc[:, ~sraw_mask].sub(
        #~ snmu).divide(snsig)
    #~ sweights_part.loc[raw_mask] = sweights_part.loc[raw_mask].divide(
        #~ snsig.loc[raw_mask]).fillna(0)
    #~ srecalc_decfun4 = (
        #~ sfeatures_part.mul(sweights_part).fillna(0).sum(1) + 
        #~ raw_sicpt_transformed + sicpt
    #~ )
    #~ raw_sicpt_transformed = -sweights_part.loc[raw_mask].mul(snmu.loc[raw_mask]
        #~ ).fillna(0).sum()
    
    # Copy because some will be changed
    features_part = features.copy()
    weights_part = weights.copy()

    # Fill all features with their session mean
    features_part = pandas.concat([
        features_part.loc[session].fillna(features_part.loc[session].mean())
        for session in features_part.index.levels[0]],
        axis=0, keys=features_part.index.levels[0]
    )

    # Normalize some of the features (the standard way) but leave the 
    # raw features alone
    features_part.loc[:, ~raw_mask] = features_part.loc[:, ~raw_mask].sub(
        mu.loc[:, ~raw_mask]).divide(
        sigma.loc[:, ~raw_mask])

    # Scale the raw weights
    weights_part.loc[:, raw_mask] = weights_part.loc[:, raw_mask].divide(
        sigma.loc[:, raw_mask]).fillna(0)

    # Transform the intercept corresponding to raw weights
    raw_icpt_component = -weights_part.loc[:, raw_mask].mul(
        mu.loc[:, raw_mask]).fillna(0).sum(1)
    icpt_transformed = raw_icpt_component + intercepts

    # Compute
    decfun = (
        features_part.mul(weights_part).fillna(0).sum(1).add(
        icpt_transformed)
    )
    
    return features_part, weights_part, icpt_transformed, decfun

def bin_features_into_analysis_bins(features_df, C2_whisk_cycles, BINS):
    """Bin locked_t by BINS and add analysis_bin as level
    
    Drops rows that are not contained by a bin
    """
    # Make a copy
    features_df = features_df.copy()
    
    # Get index as df
    idxdf = features_df.index.to_frame().reset_index(drop=True)

    # Join the peak frame
    idxdf = idxdf.join(C2_whisk_cycles['locked_t'], 
        on=['session', 'trial', 'cycle'])

    # Cut the peak frame by the frame bins
    idxdf['analysis_bin'] = pandas.cut(
        idxdf['locked_t'],
        bins=BINS['bin_edges_t'], labels=False, right=False)

    # Mark null bins with -1 (they will later be dropped)
    idxdf['analysis_bin'] = idxdf['analysis_bin'].fillna(-1).astype(np.int)
        
    # Reinsert this index
    features_df.index = pandas.MultiIndex.from_frame(
        idxdf[['session', 'trial', 'analysis_bin', 'cycle']])

    # Drop frame_bin == -1, if any
    features_df = features_df.drop(-1, level='analysis_bin')

    # Sort
    features_df = features_df.sort_index()
    
    return features_df

def load_model_results(model_dir):
    # Load results from this reduced model
    preds = pandas.read_pickle(os.path.join(model_dir, 
        'finalized_predictions')).sort_index()
    weights = pandas.read_pickle(os.path.join(model_dir, 
        'meaned_weights')).sort_index()
    intercepts = pandas.read_pickle(os.path.join(model_dir, 
        'meaned_intercepts')).sort_index()
    try:
        normalizing_mu = pandas.read_pickle(os.path.join(model_dir, 
            'big_normalizing_mu')).sort_index()
        normalizing_sigma = pandas.read_pickle(os.path.join(model_dir, 
            'big_normalizing_sigma')).sort_index()
    except FileNotFoundError:
        print("warning: obsolete filenames")
        normalizing_mu = pandas.read_pickle(os.path.join(model_dir, 
            'normalizing_mu')).sort_index()
        normalizing_sigma = pandas.read_pickle(os.path.join(model_dir, 
            'normalizing_sigma')).sort_index()        

    # Fix the way mu and sigma were stored
    # They have a redundant choice/rewside level
    normalizing_mu = normalizing_mu.xs('rewside', level=1)
    normalizing_sigma = normalizing_sigma.xs('rewside', level=1)

    # Get only session on the index, to match features
    normalizing_mu = normalizing_mu.unstack('session').T
    normalizing_sigma = normalizing_sigma.unstack('session').T

    # Also put session on the index of weights
    weights = weights.T

    # This isn't necessary
    #~ normalizing_mu = normalizing_mu.loc[:, weights.columns]
    #~ normalizing_sigma = normalizing_sigma.loc[:, weights.columns]
    assert weights.shape[1] == normalizing_mu.shape[1]
    assert weights.shape[1] == normalizing_sigma.shape[1]

    # Return
    res = {
        'preds': preds,
        'weights': weights,
        'intercepts': intercepts,
        'normalizing_mu': normalizing_mu,
        'normalizing_sigma': normalizing_sigma,
    }
    return res

def partition(model_features, model_results, raw_mask):
    """Partition the features and weights into raw and standard
    
    model_features
    
    model_results
    
    raw_mask : array of bool
        Indicates which columns (features) of the weights are raw
    
    """
    # Check standard decfun
    # Throughout, slightly different than the original decfun because we're using the 
    # mean weights instead of the fold weights
    decfun_standard = (
        recalculate_decfun_standard(
            model_features, 
            model_results['normalizing_mu'], 
            model_results['normalizing_sigma'], 
            model_results['weights'], 
            model_results['intercepts']
        )
    )

    # Check raw
    scaled_weights, icpt_transformed, decfun_raw = (
        recalculate_decfun_raw(
            model_features, 
            model_results['normalizing_mu'], 
            model_results['normalizing_sigma'], 
            model_results['weights'], 
            model_results['intercepts']
        )
    )

    # Calculate partioned
    features_part, weights_part, icpt_transformed_part, decfun_part = (
        recalculate_decfun_partitioned(
            model_features, 
            model_results['normalizing_mu'], 
            model_results['normalizing_sigma'], 
            model_results['weights'], 
            model_results['intercepts'],
            raw_mask
        )
    )

    # Error check
    # TODO: get rid of the dropna(), there shouldn't be nulls here
    assert np.allclose(decfun_raw.dropna().values, decfun_standard.dropna().values)
    assert np.allclose(decfun_raw.dropna().values, decfun_part.dropna().values)    
    
    # Return
    res = {
        'raw_mask': raw_mask,
        'features_part': features_part,
        'weights_part': weights_part,
        'icpt_transformed_part': icpt_transformed_part,
        'decfun_part': decfun_part,
    }
    return res

def iterate_behavioral_classifiers_over_targets_and_sessions(
    feature_set,
    labels,
    reg_l,
    to_optimize,
    n_splits,
    stratify_by,
    decode_targets=('rewside', 'choice'),
    verbose=True,
    min_class_size_warn_thresh=2,
    random_seed=None,
    ):
    """Runs behavioral classifier on all targets and sessions
    
    Procedure
    * For each target * session:
    *   Intifies the classes (using `intify_classes`)
    *   Stratifies (using `stratify_and_calculate_sample_weights` and
        `stratified_split_data`)
    *   Normalizes features (using `normalize_features`)
        Errors if any feature becomes >25 z-scores
    *   Checks for size of training and tuning data
    *   Runs classifier (using `tuned_logregress`)
    * Extracts the scores on the tuning set and choose best reg
    * Extracts predictions
    * Concatenates over all target and sesions
    
    verbose : bool
        If True, print each session name as it is analyzed. Also, 
        print warnings about min class ize.
    
    min_class_size_warn_thresh : int
        If `verbose` and the size of any stratification is less than
        this value, print a warning.
    
    Returns: dict
        'best_reg_by_split' 
        'scores_by_reg_and_split'
        'tuning_scores'
        'finalized_predictions'
        'best_per_row_results_by_split'
        'best_weights_by_split'
        'best_intercepts_by_split'
        'meaned_weights'
        'meaned_intercepts'
        'big_norm_session_features'
        'big_normalizing_mu'
        'big_normalizing_sigma'
    """
    # Enforce each datapoint used for tuning and testing exactly once
    # Actually this is only approximate for tuning due to complexities
    # of stratification
    splits_group_names = ['train'] * (n_splits - 2) + ['tune']
    
    
    ## Iterate over targets
    # Store results here
    # Everything is jointly optimized over all targets and sessions
    keys_l = []
    norm_session_features_l = []
    normalizing_mu_l = []
    normalizing_sigma_l = []
    tuning_keys_l = []
    tuning_results_l = []


    ## Iterate over sessions
    for session in list(feature_set.index.levels[0]):
        # Verbose
        if verbose:
            print(session)


        ## Iterate over targets
        for target in decode_targets:
        
        
            ## Select data
            # Get data for just this session, or for all sessions
            if session == 'global':
                # All sessions
                session_features = feature_set
                session_labels = labels.loc[:, target]
                session_classes = labels.loc[:, :]
            else:
                # Just this session
                session_features = feature_set.loc[session]
                session_labels = labels.loc[session, target]
                session_classes = labels.loc[session, :]


            ## Set up cross-validation
            # Code and intify classes
            intified_session_classes = intify_classes(
                session_classes, by=stratify_by)
            
            # Calculate sample weights from strats
            strat_id2weight, sample_weights = (
                stratify_and_calculate_sample_weights(
                intified_session_classes.values)
            )

            # Stratified splits into folds
            folds = stratified_split_data(
                intified_session_classes, 
                n_splits=n_splits,
                n_tune_splits=1,
                random_seed=random_seed,
                )
            
            # Check that each trial is used for testing on exactly one split
            assert ((folds == 'test').sum(1) == 1).all()


            ## Warn if too little data from any class
            # See notes in tuned_logregress about the effect of this
            size_of_each_class = (
                intified_session_classes.value_counts().sort_index())
            
            if stratify_by == ('rewside', 'choice'):
                size_of_each_class = size_of_each_class.reindex(
                    range(4)).fillna(0).astype(np.int)
            elif stratify_by in [('shape',), ('rewside',), ('choice',),]:
                size_of_each_class = size_of_each_class.reindex(
                    range(2)).fillna(0).astype(np.int)
            elif stratify_by == ('rewside', 'choice', 'servo_pos'):
                size_of_each_class = size_of_each_class.reindex(
                    range(12)).fillna(0).astype(np.int)
            elif stratify_by is None:
                size_of_each_class = size_of_each_class.reindex(
                    [0]).fillna(0).astype(np.int)
            
            if verbose and size_of_each_class.min() < min_class_size_warn_thresh:
                print(
                    "warning: some classes have fewer than {} examples".format(
                    min_class_size_warn_thresh))

            
            ## Set up features and labels
            # Normalize features
            norm_session_features, normalizing_mu, normalizing_sigma = (
                normalize_features(session_features))

            # Check for very large values
            if norm_session_features.abs().max().max() > 25:
                raise ValueError("feature got normalized too hard")

            # Intify targets
            if target == 'shape':
                intified_labels = (session_labels == 'convex').astype(np.int)
            elif target in ['rewside', 'choice']:
                intified_labels = (session_labels == 'right').astype(np.int)

            ## Tune and run
            session_tuning_keys_l, session_tuning_results_l = (
                tuned_logregress(
                folds, 
                norm_session_features, 
                intified_labels,
                sample_weights, 
                reg_l,
            ))
            session_tuning_keys_l = [tuple([session, target] + list(tup))
                for tup in session_tuning_keys_l]
            
            
            ## Store
            norm_session_features_l.append(norm_session_features)
            normalizing_mu_l.append(normalizing_mu)
            normalizing_sigma_l.append(normalizing_sigma)
            keys_l.append((session, target))

            tuning_keys_l += session_tuning_keys_l
            tuning_results_l += session_tuning_results_l
    
    
    ## Extract scores on the tuning set
    tuning_scores_l = [tuning_result['scores_df'].loc['tune', :]
        for tuning_result in tuning_results_l]
    tuning_scores = pandas.concat(tuning_scores_l, keys=tuning_keys_l, 
        axis=1, names=['session', 'decode_label', 'split', 'reg']).T    
    
    # Add a mouse column
    tuning_scores = tuning_scores.reset_index()
    tuning_scores['mouse'] = tuning_scores['session'].map(
        lambda s: s.split('_')[1])
    
    
    ## Choose the best regularization
    # For each split, choose the reg that optimizes over mouse * decode_label
    scores_by_reg_and_split = tuning_scores.groupby(
        ['mouse', 'reg', 'split'])[to_optimize].mean().mean(
        level=['reg', 'split']).unstack('split')

    # Choose best reg
    best_reg_by_split = scores_by_reg_and_split.idxmax()

    
    ## Now extract just the best reg from each split
    best_per_row_results_by_split_l = []
    best_weights_by_split_l = []
    best_intercepts_by_split_l = []
    best_keys_l = []
    
    # Iterate over all results and keep just the ones corresponding to 
    # the best_reg for that split
    for n_key, key in enumerate(tuning_keys_l):
        # Split the key
        session, decode_label, split, reg = key
        
        # Check whether this result is with the best_reg on this split
        if reg == best_reg_by_split.loc[split]:
            # Extract those results
            best_split_results = tuning_results_l[n_key]
            
            # Store
            best_per_row_results_by_split_l.append(
                best_split_results['per_row_df'])
            best_weights_by_split_l.append(
                best_split_results['weights'])
            best_intercepts_by_split_l.append(
                best_split_results['intercept'])
            best_keys_l.append((session, decode_label, split))
    
    # Concat over session, decode_label, split
    best_per_row_results_by_split = pandas.concat(
        best_per_row_results_by_split_l,
        keys=best_keys_l, 
        names=['session', 'decode_label', 'split']).sort_index()
    
    best_weights_by_split = pandas.concat(best_weights_by_split_l, 
        keys=best_keys_l, axis=1, names=['session', 'decode_label', 'split']
        ).sort_index(axis=1)

    best_intercepts_by_split = pandas.Series(best_intercepts_by_split_l,
        index=pandas.MultiIndex.from_tuples(best_keys_l, 
        names=['session', 'decode_label', 'split'])).sort_index()
    
    
    ## Finalize predictions by taking only the ones on the test set
    # Take the prediction for each row from the split where that
    # row was in the test set
    finalized_predictions = best_per_row_results_by_split.loc[
        best_per_row_results_by_split['set'] == 'test'].reset_index(
        'split').sort_index()

    # Mean the weights and intercepts over splits
    meaned_weights = best_weights_by_split.mean(
        level=['session', 'decode_label'], axis=1)
    meaned_intercepts = best_intercepts_by_split.mean(
        level=['session', 'decode_label'])
    
    
    ## Concat normalizing over sessions and decode_labels
    big_norm_session_features = pandas.concat(norm_session_features_l,
        keys=keys_l, names=['session', 'decode_label'])

    big_normalizing_mu = pandas.concat(normalizing_mu_l,
        keys=keys_l, names=['session', 'decode_label'])

    big_normalizing_sigma = pandas.concat(normalizing_sigma_l,
        keys=keys_l, names=['session', 'decode_label'])


    ## Return
    return {
        'best_reg_by_split': best_reg_by_split,
        'scores_by_reg_and_split': scores_by_reg_and_split,
        'tuning_scores': tuning_scores,
        'finalized_predictions': finalized_predictions,
        'best_per_row_results_by_split': best_per_row_results_by_split,
        'best_weights_by_split': best_weights_by_split,
        'best_intercepts_by_split': best_intercepts_by_split,
        'meaned_weights': meaned_weights,
        'meaned_intercepts': meaned_intercepts,
        'big_norm_session_features': big_norm_session_features,
        'big_normalizing_mu': big_normalizing_mu,
        'big_normalizing_sigma': big_normalizing_sigma,
        }
