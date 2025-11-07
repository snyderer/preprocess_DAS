import os
from preprocess_DAS import processing

 # !!! Naming convention I'm using: [sitename]_r[range ID], 
 # where r1 is the furthest 40km segment from the interrogator, 
 # r2 is the second furthest 40 km segment, and so on.
dataset = 'svalbard_r1'
mode = 'testing'    # processing or testing

# desired output settings (probably leave the same):
settings = {
    'fs': 200,
    'dx': 8,
    'cable_span': 40,   # km
    'use_full_cable': False,
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

interrogator = "asn_alt" # custom interrogator name for Josephine's data

sectionID = int(dataset[-1])
settings['start_distance']-=(sectionID-1)*settings['cable_span'] # correct start_distance for the specified range ID 

if mode=='testing':
    input_dir = r'C:\Users\ers334\Desktop\testingData\Svalbard_Josephine\2020' # !!! Add an input path to a smaller dataset for testing
    outDataDir = r'C:\Users\ers334\Desktop\testingData'  # !!! Add a path to a smaller dataset for testing
elif mode=='processing':
    input_dir = r'C:\Users\ers334\Desktop\testingData\Svalbard_Josephine\2020' # !!! Add an input path to a smaller dataset for testing
    outDataDir = r'H:'  # !!! where you want the data to be saved (external SSD drive path or something)

output_dir = os.path.join(outDataDir, dataset) # create an output directory where compressed data will be saved



# run data processing:
results = processing.process_directory(
    input_dir=input_dir,
    output_dir=output_dir,
    interrogator=interrogator,
    settings=settings
)

print(results)
