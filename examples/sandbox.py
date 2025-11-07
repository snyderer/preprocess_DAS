import os
from pathlib import Path
import numpy as np
import preprocess_DAS.data_io as io
import preprocess_DAS.data_formatting as df
import matplotlib.pyplot as plt

path_in = r'C:\Users\ers334\Desktop\testingData\medsea_full'

# get list of files in directory
datafiles = []
settingsfile = []
for x in os.listdir(path_in):
    if 'h5' in x:
        if 'settings' in x:
            settingsfile = Path.joinpath(Path(path_in), x)
        else:
            datafiles.append(Path.joinpath(Path(path_in), x))

# load settings:
settings = io.load_settings_preprocessed_h5(settingsfile)

iter = 0
tx_list = []
tlist = []
for file in datafiles:
    tx1, t1, x, timestamp = io.load_rehydrate_preprocessed_h5(str(file), settings)
    tlist.append(t1+iter*30)
    tx_list.append(tx1)
    iter += 1

tx = np.concatenate(tx_list, axis=1)
t = np.concatenate(tlist)
plt.figure()
plt.imshow(np.abs(tx)*1e9, extent=[t.min(), t.max(), x.min(), x.max()], aspect='auto',
    origin='lower', cmap='turbo', vmin=0, vmax=3)
plt.xlabel('time [s]')
plt.ylabel('distance [m]')
plt.title('rehydrated data')
plt.colorbar()