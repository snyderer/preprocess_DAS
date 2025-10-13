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
    DAS data loader class using DAS4Whales for consistent data access.
    
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
        
        if dw is None:
            raise ImportError("DAS4Whales is required for the Loader class")
        
        # Get file list
        self._get_file_list()
        
        # Load metadata from first file
        self._load_metadata()
        
        # Set up channel selection
        self._setup_channels(start_distance_km, cable_span_km, dx_in_m)
        
        # Create bandpass filter if specified
        self._setup_filter()
        
        # Initialize iterative loader
        self._initialize_loader()
        
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
        total_length = nx*dx

        if start_distance_km<0:
            start_distance_km = total_length/1000 + start_distance_km
        start_channel = max([int(np.floor(start_distance_km*1000/dx)), 0])

        if dx_in_m == None or dx_in_m <= self.metadata['dx']:
            step = 1
        else:
            step = int(np.floor(dx_in_m/self.metadata['dx']))

        num_channels = int(np.ceil(cable_span_km*1000/dx))
        end_channel = start_channel + num_channels

        if end_channel > nx:
            start_channel = nx-num_channels
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
    
    def _initialize_loader(self):
        """Initialize DAS4Whales iterative loader"""
        try:
            self.das_loader = dw.data_handle.iterative_loader(
                str(self.input_dir),
                self.selected_channels,
                metadata=self.metadata,
                interrogator=self.interrogator,
                start_file_index=0,  # File indexing already handled
                time_window_s=self.time_window_s
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize DAS4Whales loader: {e}")
    
    def __iter__(self):
        """Make the loader iterable"""
        return self
    
    def __next__(self):
        """Get next data chunk"""
        try:
            # Get data from DAS4Whales loader
            trace, tx, dist, timestamp = next(self.das_loader)
            
            # Apply bandpass filter if specified
            if self.filter_sos is not None:
                import scipy.signal as sp
                trace = sp.sosfiltfilt(self.filter_sos, trace, axis=1)
            
            # Create chunk info
            chunk_info = {
                'trace': trace,
                'time_axis': tx,
                'distance_axis': dist,
                'timestamp': timestamp,
                'shape': trace.shape,
                'duration': tx[-1] - tx[0] if len(tx) > 0 else 0,
                'channels': self.selected_channels,
                'filtered': self.filter_sos is not None
            }
            
            return chunk_info
            
        except StopIteration:
            raise StopIteration("No more data chunks available")
        except Exception as e:
            raise RuntimeError(f"Error loading next chunk: {e}")
    
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
            for i, file_path in enumerate(self.file_list):
                try:
                    timestamp = dw.data_handle.load_das_file_startTime(file_path, self.interrogator)
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
        # TODO: this may be complete but it still untested
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

def save_chunk_h5(filepath, fk_dehyd):
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
        
        # Original metadata group
        orig_grp = f.create_group('original_metadata')
        for key, value in original_metadata.items():
            if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                orig_grp.attrs[key] = value
            elif isinstance(value, (list, tuple)):
                orig_grp.create_dataset(key, data=np.array(value))
            elif isinstance(value, np.ndarray):
                orig_grp.create_dataset(key, data=value)
            else:
                try:
                    orig_grp.attrs[key] = str(value)
                except:
                    print(f"Warning: Could not save original metadata key '{key}'")
        
        # Processing settings group  
        proc_grp = f.create_group('processing_settings')
        for key, value in processing_settings.items():
            if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                proc_grp.attrs[key] = value
            elif isinstance(value, (list, tuple)):
                proc_grp.create_dataset(key, data=np.array(value))
            elif isinstance(value, np.ndarray):
                proc_grp.create_dataset(key, data=value)
            else:
                try:
                    proc_grp.attrs[key] = str(value)
                except:
                    print(f"Warning: Could not save processing setting '{key}'")
        
        # Rehydration info group - this is the key part!
        rehyd_grp = f.create_group('rehydration_info')
        rehyd_grp.create_dataset('nonzeros_mask', data=sample_nonzeros, 
                               compression='gzip', compression_opts=9)
        rehyd_grp.attrs['target_shape'] = sample_shape
        
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

def load_settings_h5(filepath):
    """
    Load settings and rehydration info.
    
    Returns:
    --------
    settings_data : dict
        Dictionary with all settings and rehydration info
    """
    with h5py.File(filepath, 'r') as f:
        settings_data = {}
        
        # Load original metadata
        if 'original_metadata' in f:
            orig_meta = {}
            grp = f['original_metadata']
            for key in grp.attrs:
                orig_meta[key] = grp.attrs[key]
            for key in grp.keys():
                orig_meta[key] = grp[key][...]
            settings_data['original_metadata'] = orig_meta
        
        # Load processing settings
        if 'processing_settings' in f:
            proc_settings = {}
            grp = f['processing_settings']
            for key in grp.attrs:
                proc_settings[key] = grp.attrs[key]
            for key in grp.keys():
                proc_settings[key] = grp[key][...]
            settings_data['processing_settings'] = proc_settings
        
        # Load rehydration info
        if 'rehydration_info' in f:
            rehyd_grp = f['rehydration_info']
            settings_data['rehydration_info'] = {
                'nonzeros_mask': rehyd_grp['nonzeros_mask'][...],
                'target_shape': tuple(rehyd_grp.attrs['target_shape'])
            }
        
        # Load axes
        if 'axes' in f:
            axes_grp = f['axes']
            settings_data['axes'] = {
                'frequency': axes_grp['frequency'][...],
                'wavenumber': axes_grp['wavenumber'][...]
            }
        
        return settings_data