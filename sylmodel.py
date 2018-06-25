""" Importer for syllogistic model prediction files.

Copyright 2018 Cognitive Computation Lab
University of Freiburg
Nicolas Riesterer <riestern@tf.uni-freiburg.de>
Daniel Brand <danielbrand1@gmx.de>

"""

import collections

import pandas as pd
import numpy as np

import sylhelper as sh

# List of models to include in the analysis
MODEL_WHITELIST = [
    'Atmosphere',
    'Conversion',
    'Matching',
    'PSYCOP',
    'WCS',
    'FOL',
    'FOL-Strict',
    'PHM-Min',
    'PHM-Min-Att'
]

def load_predictions(model):
    """ Loads a syllogistic model

    Parameters
    ----------
    model : str
        Name of the model to import

    Returns
    -------
    np.ndarray
        Matrix of shape (9, 64) containing the predictions of the model.

    Examples
    --------
    >>> atm = load_predictions('Atmosphere')
    >>> atm.shape
    (9, 64)
    >>> atm[:, 33]
    array([0., 0., 0., 0., 1., 1., 0., 0., 0.])

    """

    df = pd.read_csv('models/{}.csv'.format(model))
    return df.iloc[:, 1:].values

def most_frequent_model(data_df):
    syltasks = sh.syllog_tasks()
    sylconcl = sh.syllog_conclusions()

    most_freq = np.zeros((9, 64))
    for info, df in data_df.groupby('syllog'):
        cnt = sorted(dict(collections.Counter(df['conclusion'])).items(), key=lambda x: x[1], reverse=True)
        answer = cnt[0][0]
        answer_id = sylconcl.index(answer)
        syl_id = syltasks.index(info)

        # Update the most frequent answer array
        most_freq[answer_id, syl_id] = 1

    return most_freq
