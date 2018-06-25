""" General helper functions for syllogism handling.

Copyright 2018 Cognitive Computation Lab
University of Freiburg
Nicolas Riesterer <riestern@tf.uni-freiburg.de>
Daniel Brand <danielbrand1@gmx.de>

"""

def syllog_tasks():
    """ Creates a list of syllogistic task identifiers.

    Examples
    --------
    >>> tasks = syllog_tasks()
    >>> tasks[5]
    'AI2'
    >>> tasks[25]
    'IE2'
    >>> tasks[-1]
    'OO4'
    >>> len(tasks)
    64

    """

    result = []

    moods = ['A', 'I', 'E', 'O']
    for prem1 in moods:
        for prem2 in moods:
            for fig in range(1, 5):
                result.append(prem1 + prem2 + str(fig))

    return result

def syllog_conclusions():
    """ Creates a list of syllogistic conclusion identifiers.

    Examples
    --------
    >>> concls = syllog_conclusions()
    >>> concls[0]
    'Aac'
    >>> concls[5]
    'Eca'
    >>> concls[-1]
    'NVC'
    >>> len(concls)
    9

    """

    result = []

    for quantifier in ['A', 'I', 'E', 'O']:
        for direction in ['ac', 'ca']:
            result.append(quantifier + direction)
    result.append('NVC')

    return result
