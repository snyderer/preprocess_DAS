import das4whales as dw
import os
from nptdms import TdmsFile
import data_io as io
import data_formatting as df
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp

# fpath = r'\\ccb-qnap.nas.ornith.cornell.edu\CCB\projects\2022_CLOCCB_OR_S1113\Svalbard\data\090007.hdf5'
# fpath = r'\\ccb-qnap.nas.ornith.cornell.edu\CCB\projects\2022_CLOCCB_OR_S1113\MedSea\data\20230922\090007.hdf5'
# fpath = r'\\ccb-qnap.nas.ornith.cornell.edu\CCB\projects\2022_CLOCCB_OR_S1113\OOI\DASData\Silixa_DAS_South90km\OOIPacCity_UTC_20211104_172518.573.tdms'
# metadata = dw.data_handle.get_acquisition_parameters(fpath, 'silixa')
# fp = TdmsFile(fpath)


# fpath = r'E:\ooi_optasense_north_c2_r1\20211102_220001.h5'
# fpn = os.path.split(fpath)
# print(fpn)
# data = io.load_chunk_h5(fpath)
# settings = io.load_settings_h5(os.path.join(fpn[0], 'settings.h5'))

# tx = df.rehydrate(data, settings['rehydration_info']['nonzeros_mask'], settings['rehydration_info']['target_shape'])
# fs = settings['processing_settings']['fs']
# dx = settings['processing_settings']['dx']
# t = np.arange(tx.shape[1])/fs
# x = np.arange(tx.shape[0])*dx

# #plt.imshow(20*np.log10(np.abs(tx)), )

# dw.plot.plot_tx(tx, t, x)

# # load and plot original data:
# fpath_orig = r'E:\testingData\OOI\DASData\OptaSense\North_C2\North-C2-HF-P1kHz-GL30m-Sp2m-FS500Hz_2021-11-02T220001Z.h5'
# md = dw.data_handle.get_acquisition_parameters(fpath_orig, 'optasense')
# selchan = settings['processing_settings']['selected_channels']
# selchan[2] = 2
# tx_orig, t_orig, x_orig, file_begin_time_utc = dw.data_handle.load_das_data(fpath_orig, selchan, md, 'optasense')

# filter_sos = dw.dsp.butterworth_filter([5, [10, 90], 'bp'], md['fs'])
# txf_orig = sp.sosfiltfilt(filter_sos, tx_orig, axis=-1)

# fk_filter_matrix = dw.dsp.fk_filter_design(tx_orig.shape, selchan, md['dx'], md['fs'], cs_min=1400, cp_min=1450, cp_max=6800, cs_max=7000, display_filter=False)
# txf_orig = dw.dsp.fk_filter_filt(txf_orig, fk_filter_matrix)

# dw.plot.plot_tx(txf_orig, t_orig, x_orig)


tx_drive_letter = 'D'

fpath = tx_drive_letter + r':\svalbard_r2\20220822_125007.h5'
fpn = os.path.split(fpath)
print(fpn)
data = io.load_chunk_h5(fpath)
settings = io.load_settings_h5(os.path.join(fpn[0], 'settings.h5'))

tx = df.rehydrate(data, settings['rehydration_info']['nonzeros_mask'], settings['rehydration_info']['target_shape'])
fs = settings['processing_settings']['fs']
dx = settings['processing_settings']['dx']
t = np.arange(tx.shape[1])/fs
x = np.arange(tx.shape[0])*dx

#plt.imshow(20*np.log10(np.abs(tx)), )

dw.plot.plot_tx(tx, t, x)
fig = plt.gcf()
plt.text(5, 39, 'Rehydrated Data', fontsize=16, color='w', horizontalalignment='center')
plt.savefig(r'C:\Users\ers334\Documents\weeklyUpdates\2025_10_23\rehydrated_tx.png')


# load and plot original data:
fpath_orig = tx_drive_letter + r':\testingData\Svalbard\data\125007.hdf5'
md = dw.data_handle.get_acquisition_parameters(fpath_orig, 'asn')
selchan = settings['processing_settings']['selected_channels']
selchan[2] = 2
tx_orig, t_orig, x_orig, file_begin_time_utc = dw.data_handle.load_das_data(fpath_orig, selchan, md, 'asn')

filter_sos = dw.dsp.butterworth_filter([5, [10, 90], 'bp'], md['fs'])
txf_orig = sp.sosfiltfilt(filter_sos, tx_orig, axis=-1)

fk_filter_matrix = dw.dsp.fk_filter_design(tx_orig.shape, selchan, md['dx'], md['fs'], cs_min=1400, cp_min=1450, cp_max=6800, cs_max=7000, display_filter=False)
txf_orig = dw.dsp.fk_filter_filt(txf_orig, fk_filter_matrix)

dw.plot.plot_tx(txf_orig, t_orig, x_orig)
plt.text(5, 39, 'Original Data', fontsize=16, color='w', horizontalalignment='center')
plt.savefig(r'C:\Users\ers334\Documents\weeklyUpdates\2025_10_23\original_tx.png')

# move through the process step by step to see where amplitudes differences enter:
fk_mask_orig = df.create_fk_mask(tx_orig.shape, md['dx'], md['fs'])
fk_mask_orig_tapered = df.taper_mask(fk_mask_orig)

tmp = dw.dsp.hybrid_ninf_filter_design(txf_orig.shape, selchan, dx, fs, cs_min=1400., cp_min=1450., cp_max=3400, cs_max=3500, fmin=5., fmax=fs/2-5, display_filter=True)

ok=1

# txfi = df.fk_interpolate()