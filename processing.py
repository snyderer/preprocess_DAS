"""
processing.py

High-level processing functions for DAS data.
Orchestrates data loading, formatting, and saving.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import data_io as io
import data_formatting as df

def process_directory(input_dir, output_dir, interrogator, settings, 
                     start_file_index=0, end_file_index=None, verbose=True):
    """
    Process an entire directory of DAS files to F-K domain and dehydrate.
    
    Parameters:
    -----------
    input_dir : str or Path
        Directory containing input DAS files
    output_dir : str or Path
        Directory to save processed files
    interrogator : str
        Interrogator type ('optasense', 'silixa', etc.)
    settings : dict
        Processing settings containing:
        - fs: target sampling rate (Hz)
        - dx: target spatial sampling (m)
        - cable_span: cable span (km)
        - start_distance: starting distance (km, negative counts from end)
        - twin_sec: time window per chunk (s)
        - cs_min, cs_max: S-wave velocity bounds (m/s) [optional]
        - cp_min, cp_max: P-wave velocity bounds (m/s)
        - bandpass_filter: filter parameters [optional]
    start_file_index : int, optional
        Index of first file to process
    end_file_index : int, optional
        Index of last file to process (None for all files)
    verbose : bool, optional
        Whether to print progress information
    
    Returns:
    --------
    results : dict
        Processing results containing:
        - total_chunks: number of chunks processed
        - output_directory: path to output directory
        - settings_file: path to settings.h5 file
        - processing_time: total processing time
        - success: whether processing completed successfully
    """
    import time
    start_time = time.time()
    
    try:
        # Setup
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"Processing: {input_dir} -> {output_dir}")
        
        # Initialize loader
        loader = io.Loader(
            input_dir, interrogator, 
            start_distance_km=settings['start_distance'], 
            cable_span_km=settings['cable_span'], 
            dx_in_m=None,
            time_window_s=settings['twin_sec'], 
            start_file_index=start_file_index, 
            end_file_index=end_file_index, 
            bandpass_filter=settings.get('bandpass_filter')
        )
        
        # Initialize processing variables
        chunk_iter = 0
        fk_mask = None
        sample_nonzeros = None
        sample_shape = None
        f_axis = None
        k_axis = None
        
        if verbose:
            print("Starting chunk processing...")
        
        # Process each chunk
        for chunk in loader:
            trace = chunk['trace']
            timestamp = chunk['timestamp']
            
            if verbose:
                print(f"Processing chunk {chunk_iter}: {timestamp} (shape: {trace.shape})")
            
            # Interpolate to target grid
            Dfk, f_new, k_new = df.fk_interpolate(
                trace, loader.metadata['dx'], loader.metadata['fs'], 
                settings['dx'], settings['fs'], output_format='fk'
            )
            
            # Convert to positive frequencies for dehydration
            Dfk_pos = np.fft.rfft(np.fft.ifft2(np.fft.ifftshift(Dfk)), axis=1)
            Dfk_pos = np.fft.fft(Dfk_pos, axis=0)
            
            # Create F-K mask on first iteration
            if chunk_iter == 0:
                nx, nf = Dfk_pos.shape
                
                if verbose:
                    print(f"Creating F-K mask for grid size: {nx} x {nf}")
                
                fk_mask = df.create_fk_wedge_mask(
                    nx, nf, settings['dx'], settings['fs'], 
                    cs_min=settings.get('cs_min'), 
                    cp_min=settings['cp_min'], 
                    cp_max=settings['cp_max'], 
                    cs_max=settings.get('cs_max')
                )
                
                # Generate axes for settings file
                f_axis, k_axis = df.get_axes(nx, (nf-1)*2, settings['dx'], settings['fs'])
                
                if verbose:
                    mask_efficiency = np.sum(fk_mask) / np.prod(fk_mask.shape)
                    print(f"F-K mask created: {np.sum(fk_mask)}/{np.prod(fk_mask.shape)} elements kept ({mask_efficiency:.4f})")
            
            # Dehydrate
            fk_dehyd, nonzeros, shape = df.dehydrate_fk(Dfk_pos, fk_mask)
            
            # Store template info from first chunk
            if chunk_iter == 0:
                sample_nonzeros = nonzeros
                sample_shape = shape
                
                if verbose:
                    print(f"Template stored: shape={shape}, nonzeros={np.sum(nonzeros)}")
            
            # Generate output filename and save
            chunk_filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}.h5"
            outfile = output_dir / chunk_filename
            io.save_chunk_simple_h5(outfile, fk_dehyd)
            
            # Display progress
            if verbose:
                compression_ratio = len(fk_dehyd) / np.prod(shape)
                print(f"  Saved {len(fk_dehyd)} values to {chunk_filename} (compression: {compression_ratio:.4f})")
            
            chunk_iter += 1
        
        # Save settings after processing all chunks
        if verbose:
            print(f"\nProcessed {chunk_iter} chunks. Saving settings...")
        
        try:
            file_timestamps = loader.get_file_timestamps(method='first_only')
        except Exception as e:
            if verbose:
                print(f"Warning: Could not get file timestamps: {e}")
            file_timestamps = None
        
        settings_file = output_dir / "settings.h5"
        io.save_settings_h5(
            settings_file,
            loader.metadata,
            settings,
            sample_nonzeros,
            sample_shape,
            f_axis,
            k_axis,
            file_timestamps
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if verbose:
            print(f"Complete! Processed {chunk_iter} chunks in {processing_time:.1f}s")
            print(f"Settings saved to: {settings_file}")
        
        return {
            'total_chunks': chunk_iter,
            'output_directory': str(output_dir),
            'settings_file': str(settings_file),
            'processing_time': processing_time,
            'success': True
        }
        
    except Exception as e:
        if verbose:
            print(f"Error during processing: {e}")
        
        return {
            'total_chunks': 0,
            'output_directory': str(output_dir),
            'settings_file': None,
            'processing_time': time.time() - start_time,
            'success': False,
            'error': str(e)
        }