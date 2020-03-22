from builtins import zip
from builtins import range
import pandas
import my.decoders
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
    

def intify_classes(session_classes, ignore_missing=False):
    """Return integer class label for each row in `session_classes`.
    
    session_classes : DataFrame with columns rewside, choice, servo_pos
    
    ignore_missing : if True, it's okay if session_classes is missing
        some of those columns
    
    Returns : Series
    """
    # Replace each column with integers
    coded_session_classes = session_classes.replace({
        'rewside': {'left': 0, 'right': 1}, 
        'choice': {'left': 0, 'right': 1}, 
        'servo_pos': {1670: 0, 1760:1, 1850:2}
    })
    
    # Sum those integers
    if ignore_missing:
        # Initialize to zero
        intified_session_classes = pandas.Series(
            np.zeros(len(coded_session_classes), dtype=np.int),
            index=coded_session_classes.index,
            )
        
        # Add factor * each column
        if 'rewside' in coded_session_classes.columns:
            intified_session_classes += coded_session_classes['rewside']

        if 'choice' in coded_session_classes.columns:
            intified_session_classes += 2 * coded_session_classes['choice']
        
        if 'servo_pos' in coded_session_classes.columns:
            intified_session_classes += 4 * coded_session_classes['servo_pos']

    
    else:
        # Requires all present
        intified_session_classes = (
            coded_session_classes['rewside'] + 
            2 * coded_session_classes['choice'] + 
            4 * coded_session_classes['servo_pos']
        )
    
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
    shuffle=False, random_seed=None, return_arr=False, test_name='test',
    group_names=['train', 'tune']):
    """Stratify data into different groups, equalizing by class.
    
    Typically this is for splitting into "train", "tune", and "test" groups,
    while equalizing the number of rows from each stratification in each 
    group. We always want each row to occur in exactly one "test" group,
    but the relative sizes of the tune and train groups are a free parameter.
    
    The data is stratified by the values in stratifications. Each
    stratification is considered separately, and concatenated at the end.
    
    Within each stratification: the data are split into `n_splits` 
    "testing splits". Each data point will be in exactly one testing split.
    For each testing split, the remaining (non-test) data are split equally
    into each group in `group_names` (e.g., tuning and training).
    
    stratifications : Series
        index : however the data is indexed, e.g., session * trial
        values : class of that row
        If no stratification is desired, just provide the same value for each
        pandas.Series(np.ones(len(big_tm)), index=big_tm.index)
    
    group_names : list
        The names of the other datasets
        Names can be repeated in order to vary the relative sizes
        So for instance if n_splits = 5, and group_names is
        ['train', 'train', 'train', 'tune']
        Then there will be 3 parts training, 1 part tuning, and 1 part testing
        The allocation is done by modding the indices, not randomly sampling,
        so the results are always nearly exactly partitioned equally.
    
    Each row is in the testing set for exactly one split.
    Each split will have exactly the same amount of tuning and training.
    However, each row may be more or less common in the tuning and training
    sets across splits.
    
    Returns : DataFrame
        index : same as stratifications.index
        columns : split number range(n_splits)
        values : some string from [`test_name`] + group_names
    """
    # Set state
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Identify the unique values of `stratifications`
    unique_strats = np.sort(np.unique(stratifications))
    
    # These will be of length n_splits
    test_indices_by_split = [[] for n_split in range(n_splits)]
    
    # These will be of length n_splits, each of length n_groups
    group_indices_by_split = [
        [[] for n_group in range(len(group_names))]
        for n_split in range(n_splits)
    ]
    
    # Consider each stratification separately
    for strat in unique_strats:
        # Find the corresponding indices into stratifications
        indices = np.where(stratifications == strat)[0]
        
        # Shufle them
        if shuffle:
            np.random.shuffle(indices)
        
        # Equal size test splits
        split_indices = np.mod(np.arange(len(indices), dtype=np.int), n_splits)
        
        # For each test set, split the rest into groups
        for n_split in range(n_splits):
            # Get the test_indices and the rest_indices
            test_indices = indices[split_indices == n_split]
            rest_indices = indices[split_indices != n_split]

            # Split the rest_indices into groups of the appropriate sizes
            rest_indices_modded = np.mod(
                n_split + np.arange(len(rest_indices), dtype=np.int), 
                len(group_names))

            # Store the test indices
            test_indices_by_split[n_split].append(test_indices)
            
            # Store the groups
            for n_group in range(len(group_names)):
                group_indices_by_split[n_split][n_group].append(
                    rest_indices[rest_indices_modded == n_group])

    # Concatenate over strats
    for n_split in range(n_splits):
        test_indices_by_split[n_split] = np.sort(np.concatenate(
            test_indices_by_split[n_split]))
        
        for n_group in range(len(group_names)):
            group_indices_by_split[n_split][n_group] = np.sort(np.concatenate(
                group_indices_by_split[n_split][n_group]))
    
    if return_arr:
        return test_indices_by_split, group_indices_by_split
    
    # DataFrame it
    split_ser_l = []
    for n_split in range(n_splits):
        # Generate a series for this split
        split_ser = pandas.Series([''] * len(stratifications), 
            index=stratifications.index, name=n_split)
        
        # Set the test indices
        split_ser.iloc[test_indices_by_split[n_split]] = test_name
        
        # Set each group indices
        zobj = list(zip(group_names, group_indices_by_split[n_split]))
        for group_name, group_indices in zobj:
            split_ser.iloc[group_indices] = group_name
        
        # Store
        split_ser_l.append(split_ser)
    
    res = pandas.concat(split_ser_l, axis=1)
    res.columns.name = 'split'
    
    return res

def logregress2(
    features, labels, train_indices, test_indices,
    sample_weights=None, strats=None, regularization=10**5,
    testing_set_name='test', max_iter=10000, non_convergence_action='error',
    solver='lbfgs',
    ):
    """Run cross-validated logistic regression 
       
    testing_set_name : this is used to set the values in the 'set' column
        of the per_row_df, and also in scores_df
        If this is a tuning set, pass 'tune'
    
    max_iter : passed to logreg
    
    non_convergence_action : string
        if 'error': raise error when doesn't converge
        if 'pass': do nothing
    
    Returns: dict
        'weights': logreg.coef_[0],
        'intercept': logreg.intercept_[0],
        'scores_df': scores_df,
        'per_row_df': res_df,       
    """
    if len(train_indices) == 1:
        raise ValueError("must provide more than one training example")
    if len(np.unique(test_indices)) == 1:
        raise ValueError("must provide more than one label type")
    
    ## Split out test and train sets
    X_train = features[train_indices]
    X_test = features[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    
    
    ## Set up sample weights
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
        max_iter=max_iter, solver=solver
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
