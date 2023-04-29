from types import LambdaType
from scipy.special import voigt_profile
from scipy.stats import norm

def fit_function(x, mu, sigma, gamma, scale):
    return scale * voigt_profile(x - mu, sigma, gamma)
    #return scale * norm.pdf(x + LAMBDA_SHIFT, lamda, sigma)
