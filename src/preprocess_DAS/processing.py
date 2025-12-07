"""
processing.py

High-level processing functions for DAS data.
Orchestrates data loading, formatting, and saving.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import preprocess_DAS.data_io as io
import preprocess_DAS.data_formatting as df
import time
import logging
import traceback
import json

def init_logger(output_dir):
    logging.basicConfig(
        filename=output_dir + r'/processing_directory.log', 
        level=logging.ERROR,  # Only log ERROR and above
        format='%(message)s',  # Just show the message, no extra formatting
        filemode='w'
    )
    logger = logging.getLogger(__name__)
    return logger

def log_error(logger, error_text, exception, include_traceback=False):
    error_message = f"{error_text}; {str(exception)}"
    if include_traceback:
        error_message += f"\nTraceback: {traceback.format_exc()}"
    logger.error(error_message)
    return error_message

def process_directory(input_dir, output_dir, interrogator, settings, 
                      start_file_index=0, end_file_index=None, verbose=True):
    """
    Process an entire directory of DAS files to F-K domain and dehydrate.
    Supports resuming from process_state.json if processing is interrupted.
    """
    start_time = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = init_logger(str(output_dir))
    state_file = output_dir / "process_state.json"

    # --- Check for resume state ---
    resume_file_index = start_file_index
    resume_window_number = 0
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                state_data = json.load(f)
            last_proc = state_data.get("last_processed", {})
            resume_file_index = last_proc.get("file_index", start_file_index)
            resume_window_number = last_proc.get("window_number_in_file", 0)
            if verbose:
                print(f"[RESUME] Resuming from file index {resume_file_index} window {resume_window_number}")
        except Exception as e:
            if verbose:
                print(f"[RESUME] Failed to read state file: {e}. Starting from scratch.")

    try:
        if verbose:
            print(f"Processing: {input_dir} -> {output_dir}")

        # Initialize loader
        loader = io.Loader(
            input_dir, interrogator, 
            start_distance_km=settings['start_distance'], 
            cable_span_km=settings['cable_span'], 
            use_full_cable=settings['use_full_cable'],
            dx_in_m=None,
            time_window_s=settings['twin_sec'], 
            start_file_index=resume_file_index, 
            end_file_index=end_file_index, 
            bandpass_filter=settings.get('bandpass_filter')
        )

        # initialize settings.h5 (if not already extant)
        settings_h5_path = output_dir / "settings.h5"
        if not settings_h5_path.exists():
            io.save_settings_h5(
                settings_h5_path,
                loader.metadata,
                settings,
                sample_nonzeros=np.array([]),  # placeholder empty
                sample_shape=(0, 0),           # placeholder
                f_axis=np.array([]),
                k_axis=np.array([]),
                file_timestamps=[],
                file_names=[]
            )

        # Skip ahead inside loader if resuming mid-file
        if resume_window_number > 0:
            if verbose:
                print(f"Skipping {resume_window_number} windows in first file to reach resume point...")
            for _ in range(resume_window_number):
                try:
                    next(loader)
                except StopIteration:
                    break

        chunk_iter = 0
        fk_mask = None
        sample_nonzeros = None
        sample_shape = None
        f_axis = None
        k_axis = None
        file_timestamps = []
        filenames = []

        if verbose:
            print("Starting chunk processing...")

        # Process each chunk
        for chunk in loader:
            trace = chunk['trace']
            timestamp = chunk['timestamp']

            expected_samples = int(settings['twin_sec'] * loader.metadata['fs'])
            if trace.shape[1] < expected_samples:   # trace shape doesn't match expected number of samples
                if chunk.get("is_last_chunk", False): 
                    # it's the last data chunk -- too few samples to make full processing window. End processing.
                    if verbose:
                        print(f"Skipping last incomplete chunk ({trace.shape[1]} / {expected_samples}) at {timestamp}")
                    continue
                else:
                    raise ValueError(f"Chunk at {timestamp} is shorter than expected ({trace.shape[1]} / {expected_samples}). Possible bug.")

            if verbose:
                print(f"Processing chunk {chunk_iter}: {timestamp} (shape: {trace.shape})")

            try:
                # Interpolate to target FK grid
                Dfk, f_new, k_new = df.fk_interpolate(
                    trace,
                    loader.metadata['dx'],
                    loader.metadata['fs'],
                    settings['dx'],
                    settings['fs'],
                    output_format='fk',
                    pad=chunk['pad_before'],
                    chunk_timestamp=timestamp,
                    time_window_s=settings['twin_sec']
                )
                Dfk_pos = Dfk[:, len(f_new)//2 - 1:]
            except Exception as e:
                log_error(logger, f"error processing data in {timestamp}", e, include_traceback=True)
                continue

            # Make mask on first chunk
            if fk_mask is None:
                nx, nf = Dfk_pos.shape
                if verbose:
                    print(f"Creating F-K mask grid size: {nx} x {nf}")
                fk_mask = df.create_fk_mask(
                    Dfk.shape, settings['dx'], settings['fs'], 
                    cs_min=settings['cs_min'], cp_min=settings['cp_min'], 
                    cp_max=settings['cp_max'], cs_max=settings['cs_max'],
                    fmin=settings['f_min'], fmax=settings['f_max']
                )
                f_axis, k_axis = df.get_axes(nx, (nf-1)*2, settings['dx'], settings['fs'])
                if verbose:
                    print(f"Mask efficiency: {np.sum(fk_mask) / np.prod(fk_mask.shape):.4f}")

            # Dehydrate FK data
            fk_dehyd, nonzeros, shape = df.dehydrate_fk(Dfk_pos, fk_mask)

            if sample_nonzeros is None:
                sample_nonzeros = nonzeros
                sample_shape = shape
                if verbose:
                    print(f"Template stored: shape={shape}, nonzeros={np.sum(nonzeros)}")
                # update the settings.h5 to save mask:
                io.update_rehydration_info_h5(settings_h5_path, sample_nonzeros, sample_shape, f_axis, k_axis)
            
            # Save chunk
            chunk_filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}.h5"
            outfile = output_dir / chunk_filename
            file_timestamps.append(timestamp)
            filenames.append(chunk_filename)
            io.save_chunk_h5(outfile, fk_dehyd, timestamp)
            io.append_file_map_h5(settings_h5_path, timestamp, chunk_filename)
            
            # --- SAVE STATE after each chunk ---
            state_data = {
                "last_processed": {
                    "file_index": loader.file_index,                   # next file to read
                    "file_name": loader.file_list[loader.file_index-1] if loader.file_index > 0 else None,
                    "window_number_in_file": loader.window_number,     # next window in current file
                    "timestamp": timestamp.isoformat()
                },
                "total_chunks_processed": chunk_iter + 1
            }
            with open(state_file, "w") as f:
                json.dump(state_data, f)

            if verbose:
                compression_ratio = len(fk_dehyd) / np.prod(shape)

                total_files = len(loader.file_list)
                current_file_index = max(loader.file_index - 1, 0)  # last fully processed file

                # Simple percent complete based on file count
                percent_complete = (current_file_index / total_files) * 100

                # Elapsed time and ETA based on file progress
                elapsed_time_sec = time.time() - start_time
                if percent_complete > 0:
                    eta_sec = elapsed_time_sec * (100 - percent_complete) / percent_complete
                    # Convert to H:M
                    eta_hours = int(eta_sec // 3600)
                    eta_minutes = int((eta_sec % 3600) // 60)
                    eta_str = f"{eta_hours}h {eta_minutes}m remaining"
                else:
                    eta_str = "ETA unknown"

                print(f"  Saved {len(fk_dehyd)} values to {chunk_filename} "
                    f"(compression: {compression_ratio:.4f}) | "
                    f"{percent_complete:.1f}% complete, {eta_str}")

            chunk_iter += 1

        # After all chunks, save settings
        settings_file = output_dir / "settings.h5"
        io.save_settings_h5(
            settings_file,
            loader.metadata,
            settings,
            sample_nonzeros,
            sample_shape,
            f_axis,
            k_axis,
            file_timestamps,
            filenames
        )

        processing_time = time.time() - start_time
        if verbose:
            print(f"\nComplete! Processed {chunk_iter} chunks in {processing_time:.1f}s")
            print(f"Settings saved to: {settings_file}")

        return {
            'total_chunks': chunk_iter,
            'output_directory': str(output_dir),
            'settings_file': str(settings_file),
            'processing_time': processing_time,
            'success': True
        }

    except Exception as e:
        log_error(logger, 'error in main process', e, include_traceback=True)
        return {
            'total_chunks': 0,
            'output_directory': str(output_dir),
            'settings_file': None,
            'processing_time': time.time() - start_time,
            'success': False,
            'error': str(e)
        }