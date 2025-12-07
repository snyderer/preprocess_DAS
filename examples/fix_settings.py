from preprocess_DAS import data_io as io
from das4whales import data_handle as dh
import os

interrogator='optasense'
directory = r"D:\ooi_optasense_north_c3_full"
in_file = r"\\ccb-qnap.nas.ornith.cornell.edu\CCB\projects\2022_CLOCCB_OR_S1113\OOI\DASData\OptaSense\North_C3\North-C3-HF-P1kHz-GL30m-Sp2m_2021-11-02T000015Z.h5"
metadata = dh.get_acquisition_parameters(in_file, interrogator=interrogator)
processing_settings = {
    'fs': 200,
    'dx': 8,
    'cable_span' : 46.566,
    'use_full_cable': False, # if true, overrides cable_span and start_distance and uses the entire cable
    'start_distance': 20.0,    # km (if <0, counts distance from the end of cable)
    'twin_sec': 30,
    'cs_min': 1380,
    'cp_min': 1480,
    'cp_max': 6000, 
    'cs_max': 7000,
    'f_min': 10,
    'f_max': 90,
    'bandpass_filter': [5, [10, 90], 'bp'] # filter order, [lower cutoff, upper cutoff], filter type ('bp', 'hp')
}  # same settings dict from the run

fk_mask_params = {
    'cs_min': 1300,
    'cp_min': 1460,
    'cp_max': 6000,
    'cs_max': 7000,
    'f_min': 15,
    'f_max': 90
}

io.rebuild_settings_h5_from_chunks(
    settings_h5_path=os.path.join(directory, r'settings.h5'),
    chunks_dir=directory,
    metadata=metadata,
    processing_settings=processing_settings,
    dx=metadata["dx"],
    fs=metadata["fs"],
    fk_mask_params=fk_mask_params
)


