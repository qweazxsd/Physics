import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import sys
from scipy.special import erf

plt.rcParams['font.size'] = 30

fname = sys.argv[1]

data = pd.read_csv(fname).to_numpy()

theta = data[:,0]
rate = data[:,5]
rate_err = data[:,6]

x = np.linspace(theta.min(), theta.max(), 500)

fit = 9.243746058865463*(erf((0.5*0.391193989643634-x+0.008135785517220379)/(np.sqrt(2)*0.11640482432795957))\
						 -erf((-0.5*0.391193989643634-x+0.008135785517220379)/(np.sqrt(2)*0.11640482432795957)))

real_a = 21.478540539668256*(erf((0.5*0.1666-x+ 0.009344920465101915)/(np.sqrt(2)*0.15418134601521138))\
						 -erf((-0.5*0.1666-x+ 0.009344920465101915)/(np.sqrt(2)*0.15418134601521138)))

plt.errorbar(theta, rate, yerr=rate_err, ls='None', marker='.', label='Data', ms=25, elinewidth=3)
plt.plot(x, fit, label=r"$a=0.39$ rad", lw=4)
plt.plot(x, real_a, label=r'$a=0.17$ rad', lw=4)

plt.xlim((-0.5, 0.6))
plt.xlabel(r"$\theta$ [rad]")
plt.ylabel(r"Rate $\left[\frac{Counts}{Sec}\right]$")
plt.legend(loc='best')
plt.show()
