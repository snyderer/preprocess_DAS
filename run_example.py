import os
import data_io as io
import data_formatting as df
import numpy as np
import processing

settings = {
    'fs': 200,
    'dx': 8,
    'cable_span': 40,   # km
    'use_full_cable': False,
    'start_distance': -40.1,    # km (if <0, counts distance from the end of cable)
    'twin_sec': 30,
    'cs_min': 1400,
    'cp_min': 1480,
    'cp_max': 6800, 
    'cs_max': 7000,
    'bandpass_filter': [5, [10, 90], 'bp'] # filter order, [lower cutoff, upper cutoff], filter type ('bp', 'hp')
}

input_dir = r'\\ccb-qnap.nas.ornith.cornell.edu\CCB\projects\2022_CLOCCB_OR_S1113\OOI\DASData\OptaSense\North_C2'
output_dir = r'T:\OOI_optasense_North_C2'
interrogator = 'optasense'

results = processing.process_directory(
    input_dir=input_dir,
    output_dir=output_dir,
    interrogator=interrogator,
    settings=settings
)


# L = io.Loader(input_dir, interrogator, settings['start_distance'], settings['cable_span'], 0,
#               settings['twin_sec'], start_file_index=0, end_file_index=None, bandpass_filter=settings['bandpass_filter'])

# iter==0
# for chunk in L:
#     trace = chunk['trace']
#     timestamp = chunk['timestamp']
#     print(f"Loaded {timestamp}")
#     Dfk, f_new, k_new = df.fk_interpolate(trace, L.metadata['dx'], L.metadata['fs'], 
#                                         settings['dx'], settings['fs'], output_format='fk')
    
#     outfile = os.path.join(output_dir, io.get_chunk_filename(timestamp))

#     if iter==0:
#         print(f"Creating F-K mask for grid size: {Dfk.shape[0]} x {Dfk.shape[1]}")
        
#         fk_mask = df.create_fk_mask(Dfk.shape, settings['dx'], settings['fs'], 
#                                          cs_min=settings['cs_min'], 
#                                          cp_min=settings['cp_min'], 
#                                          cp_max=settings['cp_max'], 
#                                          cs_max=settings['cs_max'])
        
#         fk_dehyd, nonzeros, shape = df.dehydrate_fk(Dfk, fk_mask)
#         io.save_settings_h5(settings, L.metadata, nonzeros, shape)
#         io.save_chunk_h5(fk_dehyd, timestamp)
#     else:
#         fk_dehyd, *_ = df.dehydrate_fk(Dfk, fk_mask)
#         io.save_chunk_h5(fname, fk_dehyd)
    
#     iter += 1
#     ok=1