import os
from pathlib import Path
import numpy as np
import preprocess_DAS.data_io as io
import preprocess_DAS.data_formatting as df
import matplotlib.pyplot as plt
import das4whales.data_handle as dh
from  das4whales import dsp
import scipy.signal as sp

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
    tx1, t1, x, tstart = io.load_rehydrate_preprocessed_h5(str(file), settings)
    if iter==0:
        tstart1 = tstart
    dt = tstart-tstart1
    tlist.append(t1+dt)
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

plt.savefig(r'C:\Users\ers334\Documents\weeklyUpdates\2025_11_06\medsea_rehydrated.png')

# Compare to the original data:
orig_path_in = r'C:\Users\ers334\Desktop\testingData\MedSea\data\20230922'
orig_data_files = os.listdir(orig_path_in)
metadata = dh.get_acquisition_parameters(orig_path_in + '\\' + orig_data_files[0], 'asn')
selected_channels = [0, int(metadata['nx']-1), 2]

# design filter:
fmin=settings['processing_settings']['f_min']
fmax=metadata['fs']/2-4
bpfilt = settings['processing_settings']['bandpass_filter']
bpfilt[1] = [fmin, fmax]
filter_sos = dsp.butterworth_filter(bpfilt, metadata['fs'])


tx_list_orig = []
tlist_orig = []
iter = 0
for file in orig_data_files:
    tx1, t1, x_orig, tstart = dh.load_das_data(orig_path_in + '\\' + file, selected_channels=selected_channels, metadata=metadata, interrogator='asn')
    tx1 = sp.sosfiltfilt(filter_sos, tx1, axis=1)
    if iter==0:
        tstart1 = tstart
    dt = tstart - tstart1
    tlist_orig.append(t1+ dt.total_seconds())
    tx_list_orig.append(tx1)
    iter += 1

tx_orig = np.concatenate(tx_list_orig, axis=1)
t_orig = np.concatenate(tlist_orig)



fk_filter_matrix = df.create_fk_mask(tx_orig.shape, metadata['dx'], metadata['fs'], 
    cs_min=settings['processing_settings']['cs_min'],
    cp_min=settings['processing_settings']['cp_min'],
    cp_max=settings['processing_settings']['cp_max'],
    cs_max=settings['processing_settings']['cs_max'], 
    fmin=fmin,
    fmax=fmax, 
    return_half = False, fft_shift=True)

tx_orig = dsp.fk_filter_filt(tx_orig, fk_filter_matrix, tapering=True)

plt.figure()
plt.imshow(np.abs(tx_orig)*1e9, extent=[t_orig.min(), t_orig.max(), x_orig.min(), x_orig.max()], aspect='auto',
    origin='lower', cmap='turbo', vmin=0, vmax=1)
plt.xlabel('time [s]')
plt.ylabel('distance [m]')
plt.title('original data')
plt.colorbar()

plt.savefig(r'C:\Users\ers334\Documents\weeklyUpdates\2025_11_06\medsea_original.png')


# what happens if I fk_interpolate this?
tx_orig_fkinterp, _, _ = df.fk_interpolate(tx_orig, metadata['dx'], metadata['fs'], 
                  settings['processing_settings']['dx'],
                  settings['processing_settings']['fs'],
                  output_format = 'tx')
plt.figure()
plt.imshow(np.abs(tx_orig_fkinterp)*1e9, extent=[t_orig.min(), t_orig.max(), x_orig.min(), x_orig.max()], aspect='auto',
    origin='lower', cmap='turbo', vmin=0, vmax=1)
plt.xlabel('time [s]')
plt.ylabel('distance [m]')
plt.title('original data - interpolated')
plt.colorbar()

plt.savefig(r'C:\Users\ers334\Documents\weeklyUpdates\2025_11_06\medsea_original_fkinterp.png')

ok = 1