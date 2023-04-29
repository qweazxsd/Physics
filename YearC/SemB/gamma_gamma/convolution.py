import numpy as np
from scipy.special import erf

def fit_function(x, a, mu, sigma):
	return (erf((0.5*a+x-mu)/(np.sqrt(2)*sigma))-erf((-0.5*a+x-mu)/(np.sqrt(2)*sigma)))