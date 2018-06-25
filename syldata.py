""" Importer for syllogistic datasets.

Copyright 2018 Cognitive Computation Lab
University of Freiburg
Nicolas Riesterer <riestern@tf.uni-freiburg.de>
Daniel Brand <danielbrand1@gmx.de>

"""

import pandas as pd
import numpy as np

import sylhelper


def load_data():
    """ Loads the Ragni2016 dataset.

    Returns
    -------
    pd.DataFrame()
        Data Frame containing the columns [subj, syllog, sequence, conclusion].

    """

    return pd.read_csv('data/Ragni2016-preprocessed.csv')

def conclusion_to_idx(conclusion):
    """ Converts a conclusion string identifier to its corresponding index
    encoding.

    Parameters
    ----------
    conclusion : str
        Conclusion string encoding to convert (e.g., 'Aac').

    Returns
    -------
    int
        Integer encoding of the syllogistic conclusion ranging between 1 for
        'Aac' and 9 for 'NVC'.

    Examples
    --------
    >>> conclusion_to_idx('Aac')
    1
    >>> conclusion_to_idx('Aca')
    2
    >>> conclusion_to_idx('Iac')
    3
    >>> conclusion_to_idx('Ica')
    4
    >>> conclusion_to_idx('Oca')
    8
    >>> conclusion_to_idx('NVC')
    9

    """

    syl_conclusions = sylhelper.syllog_conclusions()
    return syl_conclusions.index(conclusion) + 1

def observation_dict(syl_df):
    """ Converts a syllogistic dataframe to the corresponding observation dict
    representation.

    """

    syl_df = syl_df.copy()

    # Prepare the data table by enabling custom sorting according to the
    # standard order of syllogisms
    syllog_tasks = sylhelper.syllog_tasks()
    syl_df['syllog'] = pd.Categorical(syl_df['syllog'], syllog_tasks)

    obs_dict = {}
    for subj_idx, subj_df in syl_df.groupby('subj'):
        conclusions = subj_df.sort_values('syllog')['conclusion'].apply(
            conclusion_to_idx).values
        obs_dict[subj_idx] = conclusions

    return obs_dict
