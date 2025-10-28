"""
data_io.py

Input/output functions for DAS data processing.
Includes loader class for DAS4Whales integration and HDF5 save/load functions.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
import das4whales as dw

class Loader:
    """
    DAS data loader class using DAS4Whales basic load function for consistent data access.
    
    This class handles initialization of metadata and provides iteration over
    a directory of DAS files with consistent preprocessing.
    """
    
    def __init__(self, input_dir, interrogator='optasense', 
                 start_distance_km=-40.01, cable_span_km=40, dx_in_m=None,
                 time_window_s=30, start_file_index=0, end_file_index=None,
                 bandpass_filter=None):
        """
        Initialize the DAS data loader.
        
        Parameters:
        -----------
        input_dir : str or Path
            Directory containing DAS files
        interrogator : str
            Interrogator type for DAS4Whales ('optasense', 'silixa', etc.)
        start_distance_km : distance from channel 0 where data will be loaded [km]
            if start_distance_km < 0, it will measure from the end of the cable 
        cable_span_km : length of cable to be loaded [km]
        dx_in_m : desired distance between cables [m]. 
            if dx_in_m < metadata['dx'], it will load every segment within defined span
        time_window_s : float
            Time window for each chunk in seconds
        start_file_index : int
            Index of first file to process
        end_file_index : int, optional
            Index of last file to process (None for all files)
        bandpass_filter : list, optional
            Filter parameters [order, [low_freq, high_freq], 'bp'] for DAS4Whales
        """
        self.input_dir = Path(input_dir)
        self.interrogator = interrogator
        self.time_window_s = time_window_s
        self.start_file_index = start_file_index
        self.end_file_index = end_file_index
        self.bandpass_filter = bandpass_filter
        
        # Get file list
        self._get_file_list()
        
        # Load metadata from first file
        self._load_metadata()
        
        # Set up channel selection
        self._setup_channels(start_distance_km, cable_span_km, dx_in_m)
        
        # Create bandpass filter if specified
        self._setup_filter()
        
        # Initialize iteration state
        self.current_file_index = 0
        self.current_data = None
        self.current_time_axis = None
        self.current_dist_axis = None
        self.current_file_timestamp = None
        self.current_time_position = 0  # Position within current file (seconds)
        
        print(f"Loader initialized:")
        print(f"  Files: {len(self.file_list)} ({self.start_file_index} to {self.end_file_index})")
        print(f"  Channels: {self.selected_channels}")
        print(f"  Sampling: {self.metadata['fs']} Hz, {self.metadata['dx']} m spacing")
        print(f"  Time window: {self.time_window_s} s")
        
    def _get_file_list(self):
        """Get and sort list of DAS files"""
        extensions = ['.h5', '.hdf5', '.tdms']
        self.file_list = []
        
        for ext in extensions:
            self.file_list.extend(self.input_dir.glob(f'*{ext}'))
        
        if not self.file_list:
            raise FileNotFoundError(f"No DAS files found in {self.input_dir}")
        
        # Sort files and convert to strings
        self.file_list = sorted([str(f) for f in self.file_list])
        
        # Apply file index limits
        if self.end_file_index is None:
            self.end_file_index = len(self.file_list)
        
        self.file_list = self.file_list[self.start_file_index:self.end_file_index]
        
        if not self.file_list:
            raise ValueError(f"No files to process in range [{self.start_file_index}:{self.end_file_index}]")
    
    def _load_metadata(self):
        """Load metadata from first file using DAS4Whales"""
        try:
            self.metadata = dw.data_handle.get_acquisition_parameters(
                self.file_list[0], 
                interrogator=self.interrogator
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata from {self.file_list[0]}: {e}")
    
    def _setup_channels(self, start_distance_km, cable_span_km, dx_in_m):
        """Set up channel selection"""
        nx = self.metadata['nx']
        dx = self.metadata['dx']
        total_length = nx * dx

        if start_distance_km < 0:
            start_distance_km = total_length / 1000 + start_distance_km
        start_channel = max([int(np.floor(start_distance_km * 1000 / dx)), 0])

        if dx_in_m is None or dx_in_m <= self.metadata['dx']:
            step = 1
        else:
            step = int(np.floor(dx_in_m / self.metadata['dx']))

        num_channels = int(np.ceil(cable_span_km * 1000 / dx))
        end_channel = start_channel + num_channels

        if end_channel > nx:
            start_channel = nx - num_channels
            end_channel = nx

        self.selected_channels = [start_channel, end_channel, step]
    
    def _setup_filter(self):
        """Create bandpass filter if specified"""
        self.filter_sos = None
        if self.bandpass_filter is not None:
            try:
                self.filter_sos = dw.dsp.butterworth_filter(
                    self.bandpass_filter, 
                    self.metadata['fs']
                )
                print(f"Bandpass filter created: {self.bandpass_filter}")
            except Exception as e:
                print(f"Warning: Failed to create filter {self.bandpass_filter}: {e}")
    
    def _load_file_data(self, file_index):
        """Load data from a specific file"""
        if file_index >= len(self.file_list):
            return False
        
        file_path = self.file_list[file_index]
        
        try:
            trace, tx, dist, timestamp = dw.data_handle.load_das_data(
                file_path,
                self.selected_channels,
                self.metadata,
                self.interrogator
            )
            
            # Apply bandpass filter if specified
            if self.filter_sos is not None:
                import scipy.signal as sp
                trace = sp.sosfiltfilt(self.filter_sos, trace, axis=1)
            
            self.current_data = trace
            self.current_time_axis = tx
            self.current_dist_axis = dist
            self.current_file_timestamp = timestamp
            self.current_time_position = 0
            
            return True
            
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return False
    
    def _get_next_chunk(self):
        """Extract next time chunk from current data"""
        if self.current_data is None:
            return None
        
        dt = 1.0 / self.metadata['fs']
        samples_per_window = int(self.time_window_s * self.metadata['fs'])
        
        # Calculate start and end sample indices
        start_sample = int(self.current_time_position * self.metadata['fs'])
        end_sample = start_sample + samples_per_window
        
        # Check if we have enough data in current file
        if start_sample >= self.current_data.shape[1]:
            return None  # No more data in this file
        
        # Adjust end sample if it exceeds available data
        end_sample = min(end_sample, self.current_data.shape[1])
        
        # Extract chunk
        trace_chunk = self.current_data[:, start_sample:end_sample]
        time_chunk = self.current_time_axis[start_sample:end_sample] 
        
        # Calculate chunk timestamp (file timestamp + time offset)
        chunk_timestamp = self.current_file_timestamp + pd.Timedelta(seconds=self.current_time_position)
        
        # Update position for next chunk
        self.current_time_position += self.time_window_s
        
        return {
            'trace': trace_chunk,
            'time_axis': time_chunk,
            'distance_axis': self.current_dist_axis,
            'timestamp': chunk_timestamp,
            'shape': trace_chunk.shape,
            'duration': time_chunk[-1] - time_chunk[0] if len(time_chunk) > 0 else 0,
            'channels': self.selected_channels,
            'filtered': self.filter_sos is not None
        }
    
    def __iter__(self):
        """Make the loader iterable"""
        # Reset to beginning
        self.current_file_index = 0
        self.current_data = None
        self.current_time_position = 0
        return self
    
    def __next__(self):
        """Get next data chunk"""
        while self.current_file_index < len(self.file_list):
            # Load current file if not already loaded
            if self.current_data is None:
                success = self._load_file_data(self.current_file_index)
                if not success:
                    # Skip to next file
                    self.current_file_index += 1
                    continue
            
            # Try to get next chunk from current file
            chunk = self._get_next_chunk()
            
            if chunk is not None:
                return chunk
            else:
                # No more chunks in current file, move to next file
                self.current_file_index += 1
                self.current_data = None
                self.current_time_position = 0
        
        # No more files to process
        raise StopIteration("No more data chunks available")
    
    def get_file_timestamps(self, method='first_only'):
        """
        Get timestamps for all files in the dataset.
        
        Parameters:
        -----------
        method : str
            'first_only': Load timestamp from first file, calculate others
            'all': Load timestamp from every file (slow)
            'filename': Extract from filenames (fast but may be inaccurate)
        
        Returns:
        --------
        timestamps : list
            List of pandas Timestamps for each file
        """
        timestamps = []
        
        if method == 'first_only':
            # Load first file timestamp and calculate others
            try:
                _, _, _, first_timestamp = dw.data_handle.load_das_data(
                    self.file_list[0],
                    [0, min(10, self.metadata['nx']), 1],
                    self.metadata,
                    self.interrogator
                )
                
                # Estimate file duration
                file_duration_s = self.metadata['ns'] / self.metadata['fs']
                
                # Calculate timestamps
                for i in range(len(self.file_list)):
                    calc_timestamp = first_timestamp + pd.Timedelta(seconds=i * file_duration_s)
                    timestamps.append(calc_timestamp)
                    
            except Exception as e:
                print(f"Warning: Could not calculate timestamps: {e}")
                timestamps = self._fallback_timestamps()
                
        elif method == 'all':
            # Load from every file (slow but accurate)
            print("Loading timestamps from all files (this may take time)...")
            for file_path in self.file_list:
                try:
                    _, _, _, timestamp = dw.data_handle.load_das_data(
                        file_path,
                        [0, min(10, self.metadata['nx']), 1],
                        self.metadata,
                        self.interrogator
                    )
                    timestamps.append(timestamp)
                except Exception as e:
                    print(f"Warning: Could not load timestamp from {file_path}: {e}")
                    # Use fallback
                    if timestamps:
                        # Estimate based on previous timestamp
                        est_duration = 3600  # 1 hour default
                        est_timestamp = timestamps[-1] + pd.Timedelta(seconds=est_duration)
                        timestamps.append(est_timestamp)
                    else:
                        timestamps.append(pd.Timestamp.now())
                        
        elif method == 'filename':
            timestamps = self._extract_timestamps_from_filenames()
            
        else:
            raise ValueError(f"Unknown timestamp method: {method}")
        
        return timestamps
    
    def _extract_timestamps_from_filenames(self):
        """Extract timestamps from filenames using common patterns"""
        import re
        timestamps = []
        
        for file_path in self.file_list:
            filename = Path(file_path).stem
            timestamp = None
            
            # Common timestamp patterns
            patterns = [
                (r'(\d{8}T\d{6})', '%Y%m%dT%H%M%S'),           # 20230115T143022
                (r'(\d{8}_\d{6})', '%Y%m%d_%H%M%S'),           # 20230115_143022  
                (r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})', '%Y-%m-%dT%H-%M-%S'),  # 2023-01-15T14-30-22
                (r'(\d{13})', None),                           # Unix timestamp ms
                (r'(\d{10})', None),                           # Unix timestamp s
            ]
            
            for pattern, fmt in patterns:
                match = re.search(pattern, filename)
                if match:
                    time_str = match.group(1)
                    try:
                        if fmt:
                            timestamp = pd.to_datetime(time_str, format=fmt)
                        elif len(time_str) == 13:  # ms
                            timestamp = pd.to_datetime(int(time_str), unit='ms')
                        elif len(time_str) == 10:  # s
                            timestamp = pd.to_datetime(int(time_str), unit='s')
                        
                        if timestamp:
                            break
                    except:
                        continue
            
            if timestamp is None:
                # Fallback to file modification time
                try:
                    mtime = Path(file_path).stat().st_mtime
                    timestamp = pd.to_datetime(mtime, unit='s')
                except:
                    timestamp = pd.Timestamp.now()
            
            timestamps.append(timestamp)
        
        return timestamps
    
    def _fallback_timestamps(self):
        """Fallback timestamp generation"""
        timestamps = []
        base_time = pd.Timestamp.now()
        
        for i in range(len(self.file_list)):
            # Space files 1 hour apart as default
            timestamp = base_time + pd.Timedelta(hours=i)
            timestamps.append(timestamp)
        
        return timestamps
    
    def get_summary(self):
        """Get summary information about the dataset"""
        return {
            'input_directory': str(self.input_dir),
            'interrogator': self.interrogator,
            'total_files': len(self.file_list),
            'file_range': (self.start_file_index, self.end_file_index),
            'selected_channels': self.selected_channels,
            'time_window_s': self.time_window_s,
            'original_sampling': {
                'fs': self.metadata['fs'],
                'dx': self.metadata['dx'], 
                'nx': self.metadata['nx'],
                'ns': self.metadata.get('ns', 'unknown')
            },
            'bandpass_filter': self.bandpass_filter,
            'first_file': self.file_list[0] if self.file_list else None,
            'last_file': self.file_list[-1] if self.file_list else None
        }

################################ \ end of Loader class #########################################

def get_chunk_filename(timestamp, extension='.h5'):
    """Generate standardized chunk filename from timestamp."""
    return f"{timestamp.strftime('%Y%m%d_%H%M%S')}{extension}"

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
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('fk_dehyd', data=fk_dehyd, 
                        compression='gzip', compression_opts=6)
        f.attrs['version'] = '1.0'
        f.attrs['n_values'] = len(fk_dehyd)
        f.create_dataset('timestamp', data=timestamp)

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
                    file_timestamps=None):
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
                if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
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
        
        # File timestamps for navigation (optional)
        if file_timestamps is not None:
            nav_grp = f.create_group('navigation')
            dt = h5py.string_dtype(encoding='utf-8')
            timestamp_strs = [pd.Timestamp(ts).isoformat() for ts in file_timestamps]
            nav_grp.create_dataset('file_start_times', data=timestamp_strs, dtype=dt)
            nav_grp.attrs['total_files'] = len(file_timestamps)

        print(f"Settings saved to {filepath}")
        print(f"  - Original metadata: {len(original_metadata)} items")
        print(f"  - Processing settings: {len(processing_settings)} items") 
        print(f"  - Rehydration template: shape={sample_shape}, nonzeros={np.sum(sample_nonzeros)}")
        if file_timestamps:
            print(f"  - File timestamps: {len(file_timestamps)} files")

def load_settings_h5(filepath):
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
        
        # Load original metadata (now all datasets)
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
                data = grp[key][...]
                # Convert single-element arrays back to scalars if appropriate
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
        
        # Load file timestamps
        if 'navigation' in f:
            nav_grp = f['navigation']
            timestamps_str = [ts.decode() if isinstance(ts, bytes) else ts 
                            for ts in nav_grp['file_start_times']]
            settings_data['file_timestamps'] = [pd.Timestamp(ts) for ts in timestamps_str]
        
        return settings_data