import das4whales as dw
import os
from nptdms import TdmsFile
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp
import preprocess_DAS.data_io as io
import preprocess_DAS.data_formatting as df

tx_data_path = r'C:\Users\ers334\Desktop\testingData'

interrogator = 'optasense'
# fpath = os.path.join(tx_data_path, r'svalbard_full\20220822_120057.h5')
# fpath = os.path.join(tx_data_path, r'medsea_full\20230922_090037.h5')
fpath = os.path.join(tx_data_path, r'C:\Users\ers334\Desktop\testingData\ooi_optasense_north_c2_full\20211102_215901.h5')
fpn = os.path.split(fpath)
print(fpn)
data = io.load_chunk_h5(fpath)
settings = io.load_settings_preprocessed_h5(os.path.join(fpn[0], 'settings.h5'))

tx = df.rehydrate(data, settings['rehydration_info']['nonzeros_mask'], settings['rehydration_info']['target_shape'])
fs = settings['processing_settings']['fs']
dx = settings['processing_settings']['dx']
t = np.arange(tx.shape[1])/fs
x = np.arange(tx.shape[0])*dx

fpath_orig = os.path.join(tx_data_path, r'OOI\DASData\OptaSense\North_C2\North-C2-HF-P1kHz-GL30m-Sp2m-FS500Hz_2021-11-02T215901Z.h5')
md = dw.data_handle.get_acquisition_parameters(fpath_orig, interrogator)
selchan = settings['processing_settings']['selected_channels']
tx_orig, t_orig, x_orig, file_begin_time_utc = dw.data_handle.load_das_data(fpath_orig, selchan, md, interrogator)
f_hi = min(90, int(md['fs']/2-5))
filter_sos = dw.dsp.butterworth_filter([5, [10, f_hi], 'bp'], md['fs'])
txf_orig = sp.sosfiltfilt(filter_sos, tx_orig, axis=-1)

fk_filter_matrix = dw.dsp.fk_filter_design(tx_orig.shape, selchan, md['dx'], md['fs'], cs_min=1400, cp_min=1450, cp_max=6800, cs_max=7000, display_filter=False)
txf_orig = dw.dsp.fk_filter_filt(txf_orig, fk_filter_matrix)

# move through the process step by step to see where amplitudes differences enter:
fk_mask_orig = df.create_fk_mask(tx_orig.shape, md['dx'], md['fs'], fmax=f_hi)

txfi, ti, xi = df.fk_interpolate(txf_orig, md['dx'], md['fs'], dx, fs, output_format='tx')

scale_factor = txfi.max()/txf_orig.max()
print(scale_factor)

# plot
v_min = 0
v_max = 9
plt.figure(figsize=(16, 6))
plt.subplot(1, 3, 1)
plt.imshow(np.abs(tx)*1e9, extent=[t.min(), t.max(), x.min(), x.max()], aspect='auto',
    origin='lower', cmap='turbo', vmin=v_min, vmax=v_max, interpolation_stage='data')
plt.xlabel('time [s]')
plt.ylabel('distance [m]')
plt.title('rehydrated data')
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(np.abs(txfi)*1e9, extent=[ti.min(), ti.max(), xi.min(), xi.max()], aspect='auto', 
    origin='lower', cmap='turbo', vmin=v_min, vmax=v_max, interpolation_stage='data')
plt.xlabel('time [s]')
plt.ylabel('distance [m]')
plt.title('original data, interpolated')
plt.colorbar()

plt.subplot(1,3,3)
plt.imshow(np.abs(np.abs(txfi)-np.abs(tx))*1e9, extent=[t_orig.min(), t_orig.max(), x_orig.min(), x_orig.max()], aspect='auto',
    origin='lower', cmap='turbo', vmin=v_min, vmax=v_max, interpolation_stage='data')
plt.xlabel('time [s]')
plt.ylabel('distance [m]')
plt.title('amplitude difference')
plt.colorbar()


# plt.show()
plt.savefig(r'C:\Users\ers334\Documents\weeklyUpdates\2025_11_06\test.png')
ok = 1