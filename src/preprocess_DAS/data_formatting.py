"""
data_formatting.py

Core functions for standardizing DAS data to consistent f-k domain representations.
Includes interpolation, dehydration, and rehydration functions.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage
from scipy.signal import windows
from scipy import signal as sp
import pandas as pd
#import das4whales as dw # only need once I start using das4whales to generate F-K mask

def fk_interpolate(data, dx, fs, new_dx, new_fs, output_format='fk',
                   pad=0, chunk_timestamp=None, time_window_s=None):
    dt = 1 / fs
    new_dt = 1 / new_fs

    if data.shape[1] == 0:
        raise ValueError("fk_interpolate received data with zero time samples.")

    # FFT to f–k space
    D = np.fft.fftshift(np.fft.fft2(data))
    nk, nt = D.shape

    k = np.fft.fftshift(np.fft.fftfreq(nk, d=dx))
    f = np.fft.fftshift(np.fft.fftfreq(nt, d=dt))

    new_nk = int(round(nk * dx / new_dx))
    new_nt_raw = int(round(nt * dt / new_dt))

    k_new = np.fft.fftshift(np.fft.fftfreq(new_nk, d=new_dx))
    f_new = np.fft.fftshift(np.fft.fftfreq(new_nt_raw, d=new_dt))

    # Anti-alias
    D[np.abs(k) > np.max(np.abs(k_new)), :] = 0
    D[:, np.abs(f) > np.max(np.abs(f_new))] = 0

    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((k, f), D, bounds_error=False, fill_value=0)
    K, F = np.meshgrid(k_new, f_new, indexing='ij')
    Dfk_full = interp(np.stack([K.ravel(), F.ravel()], axis=-1)).reshape(K.shape)
    
    scale_factor = (dx / new_dx) * (dt / new_dt)
    Dfk_full *= scale_factor

    # Phase correction for pad
    if chunk_timestamp is not None and pad > 0:
        time_shift_sec = pad * dt
        phase_shift = np.exp(-2j * np.pi * f_new * time_shift_sec)
        Dfk_full *= phase_shift[np.newaxis, :]

    # Crop
    if time_window_s is not None:
        total_samples_needed = int(round(new_fs * time_window_s))
        start_idx = int(round(pad * (new_fs / fs)))
        if start_idx >= Dfk_full.shape[1]:
            raise ValueError(f"Pad start index {start_idx} exceeds data length {Dfk_full.shape[1]}")
        end_idx = min(start_idx + total_samples_needed, Dfk_full.shape[1])
        if end_idx - start_idx <= 0:
            raise ValueError("Cropped fk_interpolate output has zero length.")
        Dfk_full = Dfk_full[:, start_idx:end_idx]
        f_new = f_new[start_idx:end_idx]

    if output_format == 'fk':
        return Dfk_full, f_new, k_new
    elif output_format == 'tx':
        tr = np.fft.ifft2(np.fft.ifftshift(Dfk_full)).real
        t = np.arange(tr.shape[1]) * new_dt
        x = np.arange(tr.shape[0]) * new_dx
        return tr, t, x

def create_fk_mask(shape, dx, fs, cs_min=1300, cp_min=1460, cp_max=6000, cs_max=7000,  
                   fmin=15, fmax=90, df_taper=14, return_half = True, fft_shift = True):
    """
    create an fk filter mask

    Parameters:
    -----------
    shape : [nx, ns], number of spatial samples by number of time samples
    dx : spatial distance [m]
    fs : sampling rate [Hz]
    cs_min, cp_min, cp_max, cs_max : range of expected soundspeeds [m/s]
    return_half : True to return only positive frequencies

    Returns:
    fk_mask : mask to be applied to F-K data
    
    TODO:  originally returned fk mask from DAS4Whales, but DAS4Whales still has a minor indexing error.
    For now, I'm running my own custom fk_filter design. I will incorporate this
    into DAS4Whales eventually. 
    """
    
    #fk_mask = dw.dsp.fk_filter_design(shape, [0, shape[0]-1, 1], dx, fs, 
    #                                cs_min=cs_min, cp_min=cp_min,
    #                                cp_max=cp_max, cs_max=cs_max)
    nx, ns = shape

    freq = np.fft.fftfreq(shape[1], d=1/fs)
    knum = np.fft.fftfreq(shape[0], d=dx)

    # design butterworth filter for taper edges:
    b, a = sp.butter(8, [fmin/(fs/2), fmax/(fs/2)], 'bp')
    _, h = sp.freqz(b, a, worN=freq, fs=fs)
    H = np.abs(h)**2
    
    fk_mask = np.tile(H, (len(knum), 1))
    
    # Apply taper to the frequencies of interest:
    fpmax = fmax + df_taper
    fpmin = fmin - df_taper

    # Find the corresponding indexes:
    fmin_idx = np.argmax(freq >= fpmin)
    fmax_idx = np.argmax(freq >= fpmax)

    # TODO I think I can vectorize this loop:
    for i in range(len(knum)):
        fs_min = np.abs(knum[i] * cs_min)
        fp_min = np.abs(knum[i] * cp_min)

        fp_max = np.abs(knum[i] * cp_max)
        fs_max = np.abs(knum[i] * cs_max)

        filter_line = np.ones(shape=[len(freq)], dtype=float, order='F')

        if fs_min != fp_min:
            # Filter transition band, ramping up from cs_min to cp_min
            selected_speed_mask = ((freq >= fs_min) & (freq <= fp_min)) # positive frequencies
            filter_line[selected_speed_mask] = np.sin(0.5 * np.pi *
                                                        (freq[selected_speed_mask] - fs_min) / (fp_min - fs_min))
            
            selected_speed_mask = ((-freq >= fs_min) & (-freq <= fp_min)) # negative frequencies
            filter_line[selected_speed_mask] = np.sin(0.5 * np.pi *
                                                        (freq[selected_speed_mask] + fs_min) / (fp_min - fs_min))
            
        if fs_max != fp_max:
            # Filter transition band, going down from cp_max to cs_max
            selected_speed_mask = ((freq >= fp_max) & (freq <= fs_max)) # positive frequencies
            filter_line[selected_speed_mask] = np.cos(0.5 * np.pi *
                                                            (freq[selected_speed_mask] - fp_max) / (fs_max - fp_max))
            
            selected_speed_mask = ((-freq >= fp_max) & (-freq <= fs_max)) # negative frequencies
            filter_line[selected_speed_mask] = np.cos(0.5 * np.pi *
                                                            (freq[selected_speed_mask] + fp_max) / (fs_max - fp_max))
        
        # Stopband
        filter_line[np.abs(freq) >= fs_max] = 0
        filter_line[np.abs(freq) < fs_min] = 0

        # Fill the filter matrix
        fk_mask[i, :] *= filter_line

    if fft_shift & return_half:
        fk_mask = fk_mask[:, :ns//2+1]
        fk_mask = np.fft.fftshift(fk_mask, axes=0)
    elif fft_shift and not return_half:
        fk_mask = np.fft.fftshift(fk_mask)
    elif not fft_shift and return_half:
        fk_mask = fk_mask[:, ns//2+1:]
    
    return fk_mask

def taper_constant(fk_mask, mask_type = 'tukey', widening_factor=.01, **kwargs):
    """
    taper the edges of a binary F-K mask by a constant "widening factor" (independent of phase speed)

     Parameters:
    -----------
    fk_mask : the mask to taper [nx x ns]
    mask_type : the type of mask/window function to apply. Options: 
        'tukey' (default)
        [others forthcoming]
    widening_factor : determines number of samples to taper over (always expands f-k mask).
        if widening_factor >= 1 it is used # of samples in taper edges
        if widening_factor < 1 it is used as portion of full image size in shortest dimension
            i.e. taper_width = int(min([nx, ns])*widening_factor)
    additional window-specific arguments can be passed as keyword arguments

    Returns:
    fk_mask_tapered : tapered mask

    example usage:
    fk_mask_tapered = taper_mask(fk_mask, 'tukey', 10, alpha=.5)

    NOTE: this is not currently used! I just kept it because it might be worth testing this vs other tapering methods at some stage.
    """
    nx, ns = fk_mask.shape

    # set taper width:
    if widening_factor>=1:
        taper_width = int(widening_factor)
    elif 0 < widening_factor < 2:
        taper_width = int(widening_factor*min([nx, ns]))
    else:
        raise ValueError("invalid widening_factor: must be > 0")
    
    if taper_width > min([nx, ns])/2:
        raise ValueError("invalid widening_factor: must be < min([nx, ns])/2")

    if mask_type.lower() == 'tukey':
        alpha = kwargs.get('alpha', 1)  # Default alpha=1 if not specified
        win = windows.tukey(2*taper_width + 1, alpha=alpha)
    else:
        raise ValueError("unrecognized mask_type")

    up_slope = win[:taper_width]
    down_slope = win[taper_width+1:]

    binary_mask = (fk_mask > 0).astype(bool)
    fk_mask_tapered = fk_mask

    # Calculate distance transforms
    distance_outside = ndimage.distance_transform_edt(~binary_mask) # calculates distance to nearest non-zero value

    outside_mask = ~binary_mask & (distance_outside <= taper_width) # locations where a taper is needed
    outside_distances = distance_outside[outside_mask].astype(int) - 1 # index of taper to be used
    fk_mask_tapered[outside_mask] = down_slope[outside_distances]

    full_row_mask = np.concatenate([up_slope, np.ones([nx-2*taper_width]), down_slope])
    full_col_mask = np.concatenate([up_slope, np.ones([ns-2*taper_width]), down_slope])
    full_image_mask = np.outer(full_row_mask, full_col_mask)
    fk_mask_tapered *= full_image_mask

    return fk_mask_tapered

def dehydrate_fk(fk_data, mask):
    """
    Dehydrate f-k domain data by applying a mask and extracting non-zero values.
    
    This function applies a spatial-frequency mask to the f-k data and stores only
    the non-zero values along with their indices for later reconstruction.
    
    Parameters:
    -----------
    fk_data : np.ndarray
        2D complex array of f-k domain data (space x positive_frequency)
    mask : np.ndarray
        2D boolean or float mask of same shape as fk_data
    
    Returns:
    --------
    fk_dehyd : np.ndarray
        1D array of non-zero f-k values
    nonzeros : np.ndarray
        Boolean mask indicating locations of non-zero values
    original_shape : tuple
        Shape tuple for reconstructing original time-space data (nx, nt)
    """
    nx, nf = fk_data.shape
    nt = (nf - 1) * 2  # Reconstruct original time samples from positive frequencies
    
    if mask.shape != fk_data.shape:
        raise ValueError(f"Mask shape {mask.shape} must match fk_data shape {fk_data.shape}")
    
    # Apply mask
    masked_fk = fk_data * mask
    
    # Get non-zero mask and extract values
    nonzeros = mask.astype(bool) if mask.dtype != bool else mask
    fk_dehyd = masked_fk[nonzeros]
    
    return fk_dehyd, nonzeros, (nx, nt)

def rehydrate(fk_dehyd, nonzeros, original_shape, return_format='tx'):
    """
    Rehydrate f-k domain data back to full representation.
    
    Parameters:
    -----------
    fk_dehyd : np.ndarray
        1D array of dehydrated f-k values
    nonzeros : np.ndarray
        Boolean mask of non-zero locations from dehydration
    original_shape : tuple
        Original time-space data shape (nx, nt)
    return_format : str
        'tx' for time-space domain, 'fk' for f-k domain
    
    Returns:
    --------
    If return_format='tx':
        tx_data : np.ndarray
            Rehydrated time-space data
    If return_format='fk':
        fk_data : np.ndarray
            Rehydrated positive-frequency f-k data
    """
    nx, nt = original_shape
    nf = nt // 2 + 1  # Number of positive frequencies
    
    if nonzeros.shape != (nx, nf):
        raise ValueError(f"Nonzeros mask shape {nonzeros.shape} inconsistent with original_shape {original_shape}")
    
    if len(fk_dehyd) != np.sum(nonzeros):
        raise ValueError(f"Length of fk_dehyd ({len(fk_dehyd)}) must match number of True values in nonzeros ({np.sum(nonzeros)})")
    
    # Reconstruct positive frequency f-k data
    fk_positive = np.zeros((nx, nf), dtype=complex)
    fk_positive[nonzeros] = fk_dehyd
    
    if return_format == 'fk':
        return fk_positive
    elif return_format == 'tx':
        # Convert back to time-space domain
        # First inverse FFT in space (full complex FFT)
        fx_domain = np.fft.ifft(fk_positive, axis=0)
        # Then inverse FFT in time (real FFT for positive frequencies)
        tx_data = np.fft.irfft(fx_domain, n=nt, axis=1)
        return tx_data
    else:
        raise ValueError("return_format must be 'tx' or 'fk'")

def dehydrate_tx(tx_data, mask):
    """
    Dehydrate time-space data by transforming to f-k domain first.
    
    This is a convenience function that combines FFT transformation with dehydration.
    
    Parameters:
    -----------
    tx_data : np.ndarray
        2D real array of time-space data (space x time)
    mask : np.ndarray
        2D mask for positive frequencies (space x positive_frequency)
    
    Returns:
    --------
    fk_dehyd : np.ndarray
        1D array of non-zero f-k values
    nonzeros : np.ndarray
        Boolean mask indicating locations of non-zero values
    original_shape : tuple
        Original shape of tx_data
    """
    nx, nt = tx_data.shape
    
    if nt % 2 != 0:
        raise ValueError("Time dimension must be even for real FFT operations")
    
    nf = nt // 2 + 1
    
    if mask.shape != (nx, nf):
        raise ValueError(f"Mask shape {mask.shape} must be (nx, nf_positive) = ({nx}, {nf})")
    
    # Transform to f-k domain (positive frequencies only)
    fk_data = np.fft.rfft(tx_data, axis=1)  # Real FFT in time
    fk_data = np.fft.fft(fk_data, axis=0)   # Complex FFT in space
    
    # Dehydrate
    return dehydrate_fk(fk_data, mask)

def get_axes(nx, nt, dx, fs):
    """
    Generate frequency and wavenumber axes for given parameters.
    
    Parameters:
    -----------
    nx : int
        Number of spatial samples
    nt : int
        Number of time samples
    dx : float
        Spatial sampling interval (m)
    fs : float
        Sampling rate (Hz)
    
    Returns:
    --------
    f_axis : np.ndarray
        Positive frequency axis (Hz)
    k_axis : np.ndarray
        Wavenumber axis (1/m), fftshifted
    """
    f_axis = np.fft.rfftfreq(nt, d=1/fs)
    k_axis = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    
    return f_axis, k_axis