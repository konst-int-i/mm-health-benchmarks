"""
Some utils for the mimic dataloader. Borrowed from
https://github.com/pliang279/MultiBench/tree/49f9be224f342de005b7f001281f33df4301eb22/robustness
"""

import numpy as np


# utils for dataloader
# Tabular
def add_tabular_noise(tests, noise_level=0.3, drop=True, swap=True):
    """
    Add various types of noise to tabular data.

    :param noise_level: Probability of randomly applying noise to each element.
    :param drop: Drop elements with probability `noise_level`
    :param swap: Swap elements with probability `noise_level`
    """

    robust_tests = np.array(tests)
    if drop:
        robust_tests = drop_entry(robust_tests, noise_level)
    if swap:
        robust_tests = swap_entry(robust_tests, noise_level)
    return robust_tests


def drop_entry(data, p):
    """
    Randomly drop elements in `data` with probability `p`

    :param data: Data to drop elements from.
    :param p: Probability of dropping elements.
    """
    for i in range(len(data)):
        for j in range(len(data[i])):
            if np.random.random_sample() < p:
                data[i][j] = 0
            else:
                data[i][j] = data[i][j]
    return data


def swap_entry(data, p):
    """
    Randomly swap adjacent elements in `data` with probability `p`.

    :param data: Data to swap elems.
    :param p: Probability of swapping elements.
    """
    for i in range(len(data)):
        for j in range(1, len(data[i])):
            if np.random.random_sample() < p:
                data[i][j] = data[i][j - 1]
                data[i][j - 1] = data[i][j]
    return data


##############################################################################
# Time-Series
def add_timeseries_noise(
    tests, noise_level=0.3, gaussian_noise=True, rand_drop=True, struct_drop=True
):
    """
    Add various types of noise to timeseries data.

    :param noise_level: Standard deviation of gaussian noise, and drop probability in random drop and structural drop
    :param gauss_noise:  Add Gaussian noise to the time series ( default: True )
    :param rand_drop: Add randomized dropout to the time series ( default: True )
    :param struct_drop: Add randomized structural dropout to the time series ( default: True )
    """
    # robust_tests = np.array(tests)
    robust_tests = tests
    if gaussian_noise:
        robust_tests = white_noise(robust_tests, noise_level)
    if rand_drop:
        robust_tests = random_drop(robust_tests, noise_level)
    if struct_drop:
        robust_tests = structured_drop(robust_tests, noise_level)
    return robust_tests


def white_noise(data, p):
    """Add noise sampled from zero-mean Gaussian with standard deviation p at every time step.

    :param data: Data to process.
    :param p: Standard deviation of added Gaussian noise.
    """
    for i in range(len(data)):
        for time in range(len(data[i])):
            data[i][time] += np.random.normal(0, p)
    return data


def random_drop(data, p):
    """Drop each time series entry independently with probability p.

    :param data: Data to process.
    :param p: Probability to drop feature.
    """
    for i in range(len(data)):
        data[i] = _random_drop_helper(data[i], p, len(np.array(data).shape))
    return data


def _random_drop_helper(data, p, level):
    """
    Helper function that implements random drop for 2-/higher-dimentional timeseris data.

    :param data: Data to process.
    :param p: Probability to drop feature.
    :param level: Dimensionality.
    """
    if level == 2:
        for i in range(len(data)):
            if np.random.random_sample() < p:
                data[i] = 0
        return data
    else:
        for i in range(len(data)):
            data[i] = _random_drop_helper(data[i], p, level - 1)
        return data


def structured_drop(data, p):
    """Drop each time series entry independently with probability p, but drop all modalities if you drop an element.

    :param data: Data to process.
    :param p: Probability to drop entire element of time series.
    """
    for i in range(len(data)):
        for time in range(len(data[i])):
            if np.random.random_sample() < p:
                data[i][time] = np.zeros(data[i][time].shape)
    return data
