"""
data_formatting.py

Core functions for standardizing DAS data to consistent f-k domain representations.
Includes interpolation, dehydration, and rehydration functions.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import das4whales as dw

def fk_interpolate(data, dx, fs, new_dx, new_fs, output_format='fk'):
    """
    Interpolates data in space and time using f-k domain interpolation.
    
    Parameters:
    -----------
    data : np.ndarray
        2D numpy array (space x time)
    dx : float
        Original spatial sampling interval (m)
    fs : float
        Original sampling rate (Hz)
    new_dx : float
        Desired spatial sampling interval (m)
    new_fs : float
        Desired sampling rate (Hz)
    output_format : str
        'fk' returns data in f-k domain, 'tx' returns data back in time-space domain
    
    Returns:
    --------
    If output_format='fk':
        Dfk : np.ndarray
            Interpolated f-k domain data
        f_new : np.ndarray
            New frequency axis
        k_new : np.ndarray
            New wavenumber axis
    If output_format='tx':
        tr : np.ndarray
            Interpolated time-space data
        t : np.ndarray
            New time axis
        x : np.ndarray
            New space axis
    """
    # FFT to f–k space
    D = np.fft.fftshift(np.fft.fft2(data))
    nk, nt = D.shape
    dt = 1/fs
    new_dt = 1/new_fs
    
    # Original axes
    k = np.fft.fftshift(np.fft.fftfreq(nk, d=dx))
    f = np.fft.fftshift(np.fft.fftfreq(nt, d=dt))
    
    # Target axes
    new_nk = int(nk * dx / new_dx)
    new_nt = int(nt * dt / new_dt)
    k_new = np.fft.fftshift(np.fft.fftfreq(new_nk, d=new_dx))
    f_new = np.fft.fftshift(np.fft.fftfreq(new_nt, d=new_dt))
    
    # Anti-aliasing: zero out frequencies/wavenumbers that can't be represented
    k_max_new = np.max(np.abs(k_new))
    f_max_new = np.max(np.abs(f_new))
    D[np.abs(k) > k_max_new, :] = 0
    D[:, np.abs(f) > f_max_new] = 0
    
    # Interpolate in f-k domain
    interp = RegularGridInterpolator((k, f), D, bounds_error=False, fill_value=0)
    K, F = np.meshgrid(k_new, f_new, indexing='ij')
    Dfk = interp(np.stack([K.ravel(), F.ravel()], axis=-1)).reshape(K.shape)
    
    if output_format == 'fk':
        return Dfk, f_new, k_new
    elif output_format == 'tx':
        tr = np.fft.ifft2(np.fft.ifftshift(Dfk)).real
        t = np.arange(tr.shape[1]) * new_dt
        x = np.arange(tr.shape[0]) * new_dx
        return tr, t, x
    else:
        raise ValueError('output_format must be "fk" or "tx"')

def create_fk_mask(nx, ns, dx, fs, cs_min=1400, cp_min=1480, cp_max=6800, cs_max=7000, return_half = True):
    """
    create an fk filter mask using das4whales
    Parameters:
    -----------
    nx = number of spatial samples
    ns = number of time samples
    dx = spatial distance
    fs = sampling rate
    cs_min, cp_min, cp_max, cs_max = range of expected soundspeeds
    return_half = True to return only positive frequencies
    """
    fk_mask = dw.dsp.fk_filter_design((nx, ns), [0, nx-1, 1], dx, fs, cs_min=cs_min, cp_min=cp_min,
                                    cp_max=cp_max, cs_max=cs_max)
    if return_half:
        fk_mask = fk_mask[:, ns//2-1:]
    return fk_mask

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

def rehydrate_fk(fk_dehyd, nonzeros, original_shape, return_format='tx'):
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
        fk_space_domain = np.fft.ifft(fk_positive, axis=0)
        # Then inverse FFT in time (real FFT for positive frequencies)
        tx_data = np.fft.irfft(fk_space_domain, n=nt, axis=1)
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