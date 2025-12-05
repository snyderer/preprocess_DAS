from preprocess_DAS import data_io as io
from pathlib import Path

filepath = Path(r"D:\ooi_optasense_north_c2_full")
h5_file = filepath / "settings.h5"
json_file = filepath / "settings.json"

io.convert_settings_h5_to_json(h5_file, json_file)