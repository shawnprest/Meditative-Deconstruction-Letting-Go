import numpy as np

def softmax_precision(dist, gamma):
    """ 
    Computes the softmax function on a set of values applying the provided precision 'gamma'
    """
    output = dist - np.max(dist, axis=0) # numerically stable softmax
    numerator = np.exp(output*gamma) # compute numerator
    softmax_output = numerator / np.sum(numerator, axis=0) # compute numerator / denominator
    return softmax_output