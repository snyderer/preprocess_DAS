import xarray as xr
import os
import pandas as pd
import matplotlib.pyplot as plt
import das4whales as dw
import numpy as np
from pyproj import Transformer
import folium 

import xarray as xr
import json

# dataset = "MedSea"
# dataset = "OOI_OptaSense_NorthC2"
# dataset = "OOI_OptaSense_SouthC1"
dataset = "svalbard"

# set default indices for find lat/lon/depth from cable position files
idx_lon = 1
idx_lat = 2
idx_depth = 3
delim = ' '

if dataset=="MedSea":
    nc_file = r'\\ccb-qnap.nas.ornith.cornell.edu\CCB\projects\2022_CLOCCB_OR_S1113\MedSea\DASPosition\map_bathy\gebco_2024_n43.3191_s42.5679_w5.6236_e6.1441.nc'
    cable_file = r'\\ccb-qnap.nas.ornith.cornell.edu\CCB\projects\2022_CLOCCB_OR_S1113\MedSea\DASPosition\MEUST_WGS84_latlondepth_corrected.txt'
    utm_zone = np.int32(-28)
    dx = 1.020188041924726  # spatial sampling in meters
elif dataset in ("OOI_OptaSense_NorthC2","OOI_OptaSense_NorthC3"):
    nc_file = r'\\ccb-qnap.nas.ornith.cornell.edu\CCB\projects\2022_CLOCCB_OR_S1113\OOI\DASPosition\map_bathy\gebco_2024_n45.9695_s44.5523_w-126.0956_e-123.5275.nc'
    cable_file = r'\\ccb-qnap.nas.ornith.cornell.edu\CCB\projects\2022_CLOCCB_OR_S1113\OOI\DASPosition\north_DAS_latlondepth.txt'
    idx_lon = 3
    idx_lat = 2
    idx_depth = 4
elif dataset in ("OOI_OptaSense_SouthC1", "Silixa_DAS_South90km"):
    nc_file = r'\\ccb-qnap.nas.ornith.cornell.edu\CCB\projects\2022_CLOCCB_OR_S1113\OOI\DASPosition\map_bathy\gebco_2024_n45.9695_s44.5523_w-126.0956_e-123.5275.nc'
    cable_file = r'\\ccb-qnap.nas.ornith.cornell.edu\CCB\projects\2022_CLOCCB_OR_S1113\OOI\DASPosition\south_DAS_latlondepth.txt'
    idx_lon = 2
    idx_lat = 1
    idx_depth = 3
    delim = '\s+'
elif dataset=="svalbard":
    nc_file = r'\\ccb-qnap.nas.ornith.cornell.edu\CCB\projects\2022_CLOCCB_OR_S1113\Svalbard\DASPosition\map_bathy\north_polar_sub_ice_2023_-1315077_-1114120_91010_429706.nc'
    cable_file = r'\\ccb-qnap.nas.ornith.cornell.edu\CCB\projects\2022_CLOCCB_OR_S1113\Svalbard\DASPosition\Svalbard_DAS_latlondepth.txt'
    idx_lon = 2
    idx_lat = 1
    idx_depth = 3
    delim = '\s+'


cable = pd.read_csv(cable_file, delimiter=delim, header=None, engine='python')

clon = cable.loc[:,idx_lon]
clat = cable.loc[:,idx_lat]
cz = cable.loc[:,idx_depth]

x, y = dw.map.latlon_to_xy(clat, clon)

# calculate distance along cable:
dist = np.cumsum(np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2))

print(cable.head)
if dataset=="svalbard":

    center_lat = np.mean(clat)
    center_lon = np.mean(clon)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles=None
    )
    folium.TileLayer(
        tiles='https://services.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
        attr='Esri Ocean',
        name='Ocean Base',
        max_zoom=13
    ).add_to(m)
    # Add your cable as a line
    cable_coords = list(zip(clat, clon))
    folium.PolyLine(
        cable_coords,
        color='red',
        weight=3,
        opacity=0.8,
        popup='DAS Cable'
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    m.show_in_browser()
    # while idx < len(cz):
    #     plt.text(cable_x[idx], cable_y[idx], str(round(dist[idx]/1000, 2))+" km")
    #     idx = idx+5000
    # plt.text(cable_x[-1], cable_y[-1], str(round(dist[-1]/1000, 2))+" km")
    # plt.show()
else:
    da = xr.open_dataset(nc_file)   # read bathymetry file
    da.elevation.plot()
    plt.plot(clon, clat)
    idx = 0
    while idx < len(cz):
        plt.text(clon.loc[idx], clat.loc[idx], str(round(dist[idx]/1000, 2))+" km")
        idx = idx+5000
    plt.text(clon[len(clat)-1], clat[len(clat)-1], str(round(dist[-1]/1000, 2))+" km")
    plt.show()
ok=1