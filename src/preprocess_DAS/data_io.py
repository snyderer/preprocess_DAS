"""
data_io.py

Input/output functions for DAS data processing.
Includes loader class for DAS4Whales integration and HDF5 save/load functions.
"""

import os
import numpy as np
import pandas as pd
import scipy.signal as sp
from pathlib import Path
import h5py
#import das4whales as dw
from das4whales import data_handle as dh, dsp
import preprocess_DAS.data_formatting as df

class Loader:
    """
    Continuous DAS data loader with optional buffer, edge padding, drift-free chunking.
    Now has automatic 'no-buffer' mode if file duration exactly matches twin_sec.
    """

    def __init__(self, input_dir, interrogator='optasense',
                 start_distance_km=-40.01, cable_span_km=40,
                 use_full_cable=False, dx_in_m=None,
                 time_window_s=30, start_file_index=0,
                 end_file_index=None, bandpass_filter=None):

        self.input_dir = Path(input_dir)
        self.interrogator = interrogator
        self.time_window_s = time_window_s
        self.start_file_index = start_file_index
        self.end_file_index = end_file_index
        self.bandpass_filter = bandpass_filter

        self._get_file_list()
        self._load_metadata()
        self._setup_channels(start_distance_km, cable_span_km, dx_in_m)
        if use_full_cable:
            self.selected_channels[0] = 0
            self.selected_channels[1] = self.metadata['nx']

        # Correct bandpass_filter settings if sampling rate is too high:
        if self.bandpass_filter is not None and self.bandpass_filter[1][1] > self.metadata['fs']:
            self.bandpass_filter[1][1] = int(np.floor(self.metadata['fs'] / 2) - 2)
        self._setup_filter()

        # ---- Decide whether to use buffer ----
        raw_samples_per_window = self.metadata['fs'] * self.time_window_s
        self.samples_per_window = int(round(raw_samples_per_window))
        self.use_buffer = not np.isclose(raw_samples_per_window, self.samples_per_window)

        # Also check that the first file duration matches desired window:
        try:
            nt_first = self.metadata.get('nt', None)  # # of samples in first file
            if nt_first is not None and nt_first != self.samples_per_window:
                self.use_buffer = True
        except Exception:
            self.use_buffer = True

        if self.use_buffer:
            self._init_padding_and_window()
        else:
            self.pad_before = 0
            self.pad_after = 0

        # --- Iteration state ---
        self.file_index = 0
        self.buffer = None
        self.buffer_time_axis = None
        self.buffer_start_timestamp = None
        self.global_start_timestamp = None
        self.global_buffer_start_sample = None
        self.dist_axis = None
        self.window_number = 0

    # ------------------------------------------------------------
    def _get_file_list(self):
        extensions = ['.h5', '.hdf5', '.tdms']
        self.file_list = []
        for ext in extensions:
            self.file_list.extend(self.input_dir.glob(f'*{ext}'))
        if not self.file_list:
            raise FileNotFoundError(f"No DAS files found in {self.input_dir}")
        self.file_list = sorted([str(f) for f in self.file_list])
        if self.end_file_index is None:
            self.end_file_index = len(self.file_list)
        self.file_list = self.file_list[self.start_file_index:self.end_file_index]

    def _load_metadata(self):
        self.metadata = dh.get_acquisition_parameters(self.file_list[0], interrogator=self.interrogator)

    def _setup_channels(self, start_distance_km, cable_span_km, dx_in_m):
        nx = self.metadata['nx']
        dx = self.metadata['dx']
        total_length = nx * dx
        if start_distance_km < 0:
            start_distance_km = total_length / 1000 + start_distance_km
        start_channel = max(int(np.floor(start_distance_km * 1000 / dx)), 0)
        step = 1 if dx_in_m is None or dx_in_m <= dx else int(np.floor(dx_in_m / dx))
        num_channels = int(np.ceil(cable_span_km * 1000 / dx))
        end_channel = start_channel + num_channels
        if end_channel > nx:
            start_channel = nx - num_channels
            end_channel = nx
        self.selected_channels = [start_channel, end_channel, step]

    def _setup_filter(self):
        self.filter_sos = None
        if self.bandpass_filter is not None:
            self.filter_sos = dsp.butterworth_filter(self.bandpass_filter, self.metadata['fs'])

    def _init_padding_and_window(self):
        fs = self.metadata['fs']
        raw_samples_per_window = fs * self.time_window_s
        self.samples_per_window = int(np.floor(raw_samples_per_window))
        fractional = raw_samples_per_window - self.samples_per_window
        extra_pad_samples = int(np.ceil(fractional) + 1)  # at least 1 sample
        self.pad_before = extra_pad_samples
        self.pad_after = extra_pad_samples

    def _load_file_data(self, file_index):
        trace, tx, dist, timestamp = dh.load_das_data(
            self.file_list[file_index], self.selected_channels, self.metadata, self.interrogator
        )
        trace = trace.astype(np.float32, copy=False)
        if self.filter_sos is not None:
            trace = sp.sosfiltfilt(self.filter_sos, trace, axis=1)
        if self.buffer is None:
            self.buffer = trace
            self.buffer_time_axis = tx
            self.buffer_start_timestamp = timestamp
            self.dist_axis = dist
        else:
            self.buffer = np.concatenate((self.buffer, trace), axis=1)
            self.buffer_time_axis = np.concatenate((self.buffer_time_axis, tx))
        return True

    # ------------------------------------------------------------
    # No-buffer mode: just read one file and return as a chunk
    def _get_next_whole_file_chunk(self):
        if self.file_index >= len(self.file_list):
            return None
        trace, tx, dist, timestamp = dh.load_das_data(
            self.file_list[self.file_index], self.selected_channels, self.metadata, self.interrogator
        )
        if self.filter_sos is not None:
            trace = sp.sosfiltfilt(self.filter_sos, trace, axis=1)
        self.file_index += 1
        return {
            "trace": trace,
            "time_axis": tx,
            "distance_axis": dist,
            "timestamp": timestamp,
            "channels": self.selected_channels,
            "filtered": self.filter_sos is not None,
            "pad_before": 0,
            "pad_after": 0
        }

    # Buffer mode: original continuous chunk logic
    def _get_next_continuous_chunk(self):
        fs = self.metadata['fs']

        if self.global_start_timestamp is None:
            while self.file_index < len(self.file_list) and self.buffer is None:
                if self._load_file_data(self.file_index):
                    self.file_index += 1
                else:
                    self.file_index += 1
            if self.buffer is None:
                return None
            self.global_start_timestamp = self.buffer_start_timestamp
            self.global_buffer_start_sample = 0
            self.window_number = 0

        start_sample = self.window_number * self.samples_per_window
        end_sample = start_sample + self.samples_per_window + self.pad_before + self.pad_after

        rel_start = start_sample - self.global_buffer_start_sample
        rel_end = end_sample - self.global_buffer_start_sample

        while rel_end > self.buffer.shape[1]:
            if self.file_index >= len(self.file_list):
                if self.buffer.shape[1] > rel_start + self.samples_per_window:
                    rel_end = min(rel_end, self.buffer.shape[1])
                    break
                else:
                    return None
            if self._load_file_data(self.file_index):
                self.file_index += 1
            else:
                self.file_index += 1

        if rel_start >= self.buffer.shape[1]:
            return None

        rel_end = min(rel_end, self.buffer.shape[1])

        chunk_trace = self.buffer[:, rel_start:rel_end]
        chunk_time_axis = self.buffer_time_axis[rel_start:rel_end]
        chunk_timestamp = self.global_start_timestamp + pd.Timedelta(seconds=start_sample / fs)
        self.window_number += 1

        next_start_sample = self.window_number * self.samples_per_window
        retain_from_absolute = max(0, next_start_sample - self.pad_before)
        drop_count = retain_from_absolute - self.global_buffer_start_sample
        if drop_count > 0:
            self.buffer = self.buffer[:, drop_count:]
            self.buffer_time_axis = self.buffer_time_axis[drop_count:]
            self.global_buffer_start_sample = retain_from_absolute

        return {
            "trace": chunk_trace,
            "time_axis": chunk_time_axis,
            "distance_axis": self.dist_axis,
            "timestamp": chunk_timestamp,
            "channels": self.selected_channels,
            "filtered": self.filter_sos is not None,
            "pad_before": self.pad_before,
            "pad_after": self.pad_after
        }

    # ------------------------------------------------------------
    def __iter__(self):
        self.file_index = 0
        self.buffer = None
        self.buffer_time_axis = None
        self.buffer_start_timestamp = None
        self.global_start_timestamp = None
        self.global_buffer_start_sample = None
        self.dist_axis = None
        self.window_number = 0
        return self

    def __next__(self):
        if self.use_buffer:
            chunk = self._get_next_continuous_chunk()
        else:
            chunk = self._get_next_whole_file_chunk()
        if chunk is None:
            raise StopIteration
        return chunk
    
################################ \ end of Loader class #########################################

def save_chunk_h5(filepath, fk_dehyd, timestamp):
    """
    Save only the essential dehydrated F-K data.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to save chunk file (timestamp should be in filename)
    fk_dehyd : np.ndarray
        1D array of dehydrated F-K values
    """
    posix_ts = pd.Timestamp(timestamp).timestamp()
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('fk_dehyd', data=fk_dehyd, 
                        compression='gzip', compression_opts=6)
        f.attrs['version'] = '1.0'
        f.attrs['n_values'] = len(fk_dehyd)
        f.create_dataset('timestamp', data=posix_ts)

def load_chunk_h5(filepath):
    """
    Load dehydrated F-K data from simple chunk file.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to chunk file
    
    Returns:
    --------
    fk_dehyd : np.ndarray
        1D array of dehydrated F-K values
    """
    with h5py.File(filepath, 'r') as f:
        return f['fk_dehyd'][...]

def save_settings_h5(filepath, original_metadata, processing_settings, 
                    sample_nonzeros, sample_shape, f_axis, k_axis,
                    file_timestamps=None, file_names=None):
    """
    Save all settings and rehydration info to settings.h5.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to save settings.h5 file
    original_metadata : dict
        Original metadata from DAS4Whales
    processing_settings : dict
        Processing parameters used
    sample_nonzeros : np.ndarray
        Boolean mask template for rehydration
    sample_shape : tuple
        Target shape after interpolation (nx, nt)
    f_axis : np.ndarray
        Frequency axis for target grid
    k_axis : np.ndarray  
        Wavenumber axis for target grid
    file_timestamps : list, optional
        List of file start timestamps for navigation
    file_names: list of strings representing filenames matching each file_timestamp
    """
    with h5py.File(filepath, 'w') as f:
        # Root attributes
        f.attrs['created'] = pd.Timestamp.now().isoformat()
        f.attrs['version'] = '1.0'
        f.attrs['software'] = 'DAS F-K Processor'
        
        # Original metadata group - save everything as datasets
        orig_grp = f.create_group('original_metadata')
        for key, value in original_metadata.items():
            try:
                if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                    # Save scalars as single-element datasets instead of attributes
                    orig_grp.create_dataset(key, data=value)
                elif isinstance(value, (list, tuple)):
                    orig_grp.create_dataset(key, data=np.array(value))
                elif isinstance(value, np.ndarray):
                    orig_grp.create_dataset(key, data=value)
                else:
                    # Convert to string and save as dataset
                    orig_grp.create_dataset(key, data=str(value))
            except Exception as e:
                print(f"Warning: Could not save original metadata key '{key}': {e}")
        
        # Processing settings group - save everything as datasets
        proc_grp = f.create_group('processing_settings')
        for key, value in processing_settings.items():
            try:
                if 'filter' in key:
                    bp_grp = proc_grp.create_group('bandpass_filter')
                    bp_grp.create_dataset('filter_order', data=value[0])
                    bp_grp.create_dataset('cutoff_freqs', data=np.array(value[1]))  # [fc_low, fc_hi]
                    dt = h5py.string_dtype(encoding='utf-8')
                    bp_grp.create_dataset('filter_type', data=value[2], dtype=dt)
                elif isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                    # Save scalars as single-element datasets instead of attributes
                    proc_grp.create_dataset(key, data=value)
                elif isinstance(value, (list, tuple)):
                    proc_grp.create_dataset(key, data=np.array(value))
                elif isinstance(value, np.ndarray):
                    proc_grp.create_dataset(key, data=value)
                else:
                    # Convert to string and save as dataset
                    proc_grp.create_dataset(key, data=str(value))
            except Exception as e:
                print(f"Warning: Could not save processing setting '{key}': {e}")
        
        # Rehydration info group - this should definitely be created!
        rehyd_grp = f.create_group('rehydration_info')
        rehyd_grp.create_dataset('nonzeros_mask', data=sample_nonzeros, 
                               compression='gzip', compression_opts=9)
        rehyd_grp.create_dataset('target_shape', data=np.array(sample_shape))
        rehyd_grp.attrs['description'] = 'Template for rehydrating processed chunks'
        
        # Axes group
        axes_grp = f.create_group('axes')
        axes_grp.create_dataset('frequency', data=f_axis, compression='gzip')
        axes_grp['frequency'].attrs['units'] = 'Hz'
        axes_grp.create_dataset('wavenumber', data=k_axis, compression='gzip')
        axes_grp['wavenumber'].attrs['units'] = '1/m'
        
        # File timestamps for file mapping (optional)
        if file_timestamps is not None:
            if 'file_names' not in locals():
                raise ValueError("Need to provide file_names list along with file_timestamps")

            # Define structured dtype: timestamp as float64, filename as UTF-8 string
            dt = np.dtype([
                ('timestamp', 'f8'),  # POSIX seconds
                ('filename', h5py.string_dtype(encoding='utf-8'))
            ])

            # Build structured array from provided lists
            table_data = np.array([
                (pd.Timestamp(ts).timestamp(), fname)
                for ts, fname in zip(file_timestamps, file_names)
            ], dtype=dt)

            # Save at root level as 'file_map'
            f.create_dataset('file_map', data=table_data)
            f['file_map'].attrs['total_files'] = len(table_data)

        print(f"Settings saved to {filepath}")
        print(f"  - Original metadata: {len(original_metadata)} items")
        print(f"  - Processing settings: {len(processing_settings)} items") 
        print(f"  - Rehydration template: shape={sample_shape}, nonzeros={np.sum(sample_nonzeros)}")
        if file_timestamps:
            print(f"  - File timestamps: {len(file_timestamps)} files")

def load_settings_preprocessed_h5(filepath):
    """
    Load settings and rehydration info.
    
    Returns:
    --------
    settings_data : dict
        Dictionary with all settings and rehydration info
    """
    with h5py.File(filepath, 'r') as f:
        settings_data = {
            'created': f.attrs.get('created', 'unknown'),
            'version': f.attrs.get('version', 'unknown')
        }
        
        # Load original metadata
        if 'original_metadata' in f:
            orig_meta = {}
            grp = f['original_metadata']
            
            for key in grp.keys():
                data = grp[key][...]
                # Convert single-element arrays back to scalars if appropriate
                if isinstance(data, np.ndarray) and data.size == 1 and data.dtype.kind in 'biufc':
                    orig_meta[key] = data.item()
                elif isinstance(data, bytes):
                    orig_meta[key] = data.decode()
                else:
                    orig_meta[key] = data
            
            settings_data['original_metadata'] = orig_meta
        
        # Load processing settings (now all datasets)
        if 'processing_settings' in f:
            proc_settings = {}
            grp = f['processing_settings']
            
            for key in grp.keys():
                if isinstance(grp[key], h5py.Group):  # it's a subgroup
                    if key == 'bandpass_filter':
                        bp_grp = grp['bandpass_filter']
                        filter_order = bp_grp['filter_order'][()]
                        cutoff_freqs = bp_grp['cutoff_freqs'][...]
                        filter_type = bp_grp['filter_type'][()].decode() \
                            if isinstance(bp_grp['filter_type'][()], bytes) else bp_grp['filter_type'][()]
                        proc_settings['bandpass_filter'] = [filter_order, list(cutoff_freqs), filter_type]
                else:
                    data = grp[key][...]
                    if isinstance(data, np.ndarray) and data.size == 1 and data.dtype.kind in 'biufc':
                        proc_settings[key] = data.item()
                    elif isinstance(data, bytes):
                        proc_settings[key] = data.decode()
                    else:
                        proc_settings[key] = data
            
            settings_data['processing_settings'] = proc_settings
        
        # Load rehydration info
        if 'rehydration_info' in f:
            rehyd_grp = f['rehydration_info']
            settings_data['rehydration_info'] = {
                'nonzeros_mask': rehyd_grp['nonzeros_mask'][...],
                'target_shape': tuple(rehyd_grp['target_shape'][...])
            }
        
        # Load axes
        if 'axes' in f:
            axes_grp = f['axes']
            settings_data['axes'] = {
                'frequency': axes_grp['frequency'][...],
                'wavenumber': axes_grp['wavenumber'][...]
            }
        
        # Load file_map (structured array)
        if 'file_map' in f:
            table = f['file_map'][...]  # structured array
            timestamps = table['timestamp']  # numpy array of floats
            filenames = table['filename']    # numpy array of strings
            # Convert to list of (datetime, filename)
            settings_data['file_map'] = [
                (pd.Timestamp.fromtimestamp(ts), fname) for ts, fname in zip(timestamps, filenames)
            ]
        return settings_data

def load_preprocessed_h5(filepath):
    """
    Just loads a preprocessed h5 file in its compressed format.
    """
    with h5py.File(filepath, 'r') as h:
        fk_dehyd = h['fk_dehyd'][...]
        timestamp = h['timestamp'][()]
    return fk_dehyd, timestamp

def load_rehydrate_preprocessed_h5(filepath, settings_data = None, return_format = 'tx'):
    """
    Loads and rehydrates preprocessed data from a selected h5 file
    """
    if settings_data is None:
        # find the settings file in this directory:
        settings_file = os.path.dirname(filepath) + r'\settings.h5'
        settings_data = load_settings_preprocessed_h5(settings_file)
    
    fk_dehyd, timestamp = load_preprocessed_h5(filepath)
    data = df.rehydrate(fk_dehyd, settings_data['rehydration_info']['nonzeros_mask'], settings_data['rehydration_info']['target_shape'], return_format)

    if return_format.lower() == 'tx':
        t = np.arange(data.shape[1])/settings_data['processing_settings']['fs']
        x = np.arange(data.shape[0])*settings_data['processing_settings']['dx']
        return data, t, x, timestamp
    elif return_format.lower() == 'fk':
        f = np.fft.rfftfreq(data.shape[1], d=1/settings_data['processing_settings']['fs'])
        k = np.fft.fftshift(np.fft.fftfreq(data.shape[0], d=settings_data['processing_settings']['dx']))
        return data, f, k, timestamp