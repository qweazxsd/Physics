
import numpy as np

def fit_function(x, mu1, sigma1, scale1, mu2, sigma2, scale2):
    return scale1*np.exp(-1/2 * ((x-mu1)/sigma1)**2) + scale2*np.exp(-1/2 * ((x-mu2)/sigma2)**2)
