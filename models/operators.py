import numpy as np


def andneuron(fuzzy_output, weights):
    # Implements AND logic here
    return np.prod(np.array(fuzzy_output) * weights)


def orneuron(fuzzy_output, weights):
    # Implements OR logic here
    return (
        np.sum(np.array(fuzzy_output) * weights) / np.sum(weights)
        if np.sum(weights) > 0
        else 0
    )