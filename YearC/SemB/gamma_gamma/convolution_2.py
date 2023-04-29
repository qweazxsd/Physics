import numpy as np
from scipy.special import erf
from scipy import signal

def rect(x, a):
	return np.heaviside(a/2 - x, 0.5) - np.heaviside(-a/2 - x, 0.5)

def fit_function(x, a, mu, sigma, scale):
	#return scale*np.convolve(np.exp(-1/2 * ((x-mu)/sigma)**2), rect(x, a), 'same' )
	return scale*(signal.convolve(np.exp(-1/2 * ((x-mu)/sigma)**2), rect(x, a), mode='same') / sum(rect(x, a)))