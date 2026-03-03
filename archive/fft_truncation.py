#%%
import numpy as np
import matplotlib.pyplot as plt
from lablib102.core import _fdata_keep_n_lowfreq_pnts
# by the magic of a __init__.py layer for direct core.py importation, what won't work is: from lablib102 import _fdata_keep_n_lowfreq_pnts
from scipy.fft import fft, ifft

data = np.random.randn(8)
fdata = fft(data)
ifdata = ifft(fdata)
fdata_filtered = _fdata_keep_n_lowfreq_pnts(fdata, 3)
ifdata_filtered = ifft(fdata_filtered)

fig, ax = plt.subplots()
for id, e in enumerate([data, np.abs(fdata), ifdata.real, ifdata_filtered.real]):
    ax.plot(e, ls = ":" if id ==3 else None, color = "k" if id ==0 else None)
# %%
