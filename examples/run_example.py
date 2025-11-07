import os
from preprocess_DAS import data_io as io
from preprocess_DAS import data_formatting as df
from preprocess_DAS import processing
import numpy as np

dataset = 'medsea_full'
# dataset = 'ooi_optasense_north_c2_full'
mode = 'testing'    # processing or testing

settings = {
    'fs': 200,
    'dx': 8,
    'cable_span': 40,   # km
    'use_full_cable': True, # if true, overrides cable_span and start_distance and uses the entire cable
    'start_distance': -40.01,    # km (if <0, counts distance from the end of cable)
    'twin_sec': 30,
    'cs_min': 1380,
    'cp_min': 1480,
    'cp_max': 6000, 
    'cs_max': 7000,
    'f_min': 15,
    'f_max': 90,
    'bandpass_filter': [5, [10, 90], 'bp'] # filter order, [lower cutoff, upper cutoff], filter type ('bp', 'hp')
}
if mode=='testing':
    rootDataDir = r'C:\Users\ers334\Desktop\testingData'
    outDataDir = r'C:\Users\ers334\Desktop\testingData'
elif mode=='processing':
    rootDataDir = r'\\ccb-qnap.nas.ornith.cornell.edu\CCB\projects\2022_CLOCCB_OR_S1113'
    outDataDir = r'H:'

if "full" not in dataset.lower(): # a range has been defined
    sectionID = int(dataset[-1])
    settings['start_distance']-=(sectionID-1)*settings['cable_span']

if 'svalbard' in dataset.lower():
    input_dir = os.path.join(rootDataDir, r'Svalbard\data')
    output_dir = os.path.join(outDataDir, dataset)
    interrogator = 'asn'
elif 'ooi_optasense_north' in dataset.lower():
    if 'c2' in dataset: 
        cnum = 2
    elif 'c3' in dataset:
        cnum = 3
    input_dir = os.path.join(rootDataDir, r'OOI\DASData\OptaSense\North_C'+str(cnum))
    output_dir = os.path.join(outDataDir, dataset)
    interrogator='optasense'
elif 'ooi_optasense_south' in dataset.lower():
    if 'c2' in dataset: 
        cnum = 2
    elif 'c3' in dataset:
        cnum = 3
    input_dir = os.path.join(rootDataDir, r'OOI\DASData\OptaSense\South_C'+str(cnum))
    output_dir = os.path.join(outDataDir, dataset)
    interrogator='optasense'
elif 'medsea' in dataset.lower():
    input_dir = os.path.join(rootDataDir, r'MedSea\data\20230922')
    output_dir = os.path.join(outDataDir, dataset)
    interrogator = 'asn'

# run data processing:
results = processing.process_directory(
    input_dir=input_dir,
    output_dir=output_dir,
    interrogator=interrogator,
    settings=settings
)

print(results)
