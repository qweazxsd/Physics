import numpy as np

def fit_function(x, mu, sigma, scale):
    return scale*np.exp(-1/2 * ((x-mu)/sigma)**2)
