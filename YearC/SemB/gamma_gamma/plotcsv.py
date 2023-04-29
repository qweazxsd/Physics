import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import sys
from scipy.special import voigt_profile
from scipy.signal import chirp, find_peaks, peak_widths

plt.rcParams['font.size'] = 30

fname = sys.argv[1]

data = pd.read_csv(fname).to_numpy()

channels = data[:,0]
rate = data[:,1]
rate_err = data[:,2]



#plt.errorbar(channels, rate, yerr=rate_err, ls='None', marker='.')
#plt.plot(channels, rate, ls='None', marker='.', ms=5)
plt.plot(channels, rate, lw=2, label='Data')

# na fit
#y = 417.13804358208165*np.exp(-1/2 * ((channels-1301.9243592560658)/80.53063735293917)**2) \
#	+ 37.70267170342119*np.exp(-1/2 * ((channels-3170.3910235690837)/189.03296848431663)**2)

# co fit
y = 90.50247447278781*np.exp(-1/2 * ((channels-2844.9271565060676)/111.12390298570489)**2) \
	+ 85.03519843478834*np.exp(-1/2 * ((channels-3240.931121181918)/195.90278393367916)**2)


plt.plot(channels, y,lw=4, label="Fit")
plt.xlim((0, 4500))
plt.xlabel("Channel Number")
plt.ylabel(r"Rate $\left[\frac{Counts}{Sec}\right]$")
plt.legend(loc='best')
plt.show()
