import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fname = sys.argv[1]
time = float(sys.argv[2])

nchannels = 2**13
counts = np.empty(nchannels)
channels = np.arange(nchannels) + 1

with open(fname, 'r') as f:
    for i, line in enumerate(f):
        counts[i] = int(line)

rate = counts/time
rate_err = np.sqrt(counts)/time
pd.DataFrame({'rate': rate, 'rate_err': rate_err}).to_csv(fname[:-4]+'.csv', header=True)
