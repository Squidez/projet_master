import os
import random
from PIL import Image

from noise_modif import apply_noise_modif
from wave_modif import apply_wave_modif
from brightness_modif import apply_brightness_modif

FLAG_LIST_DIR = 'flag_list.txt'
DATASET_DIR = 'dataset'

with open(FLAG_LIST_DIR, 'r') as f:
    flags = [line[:-1] for line in f]

selected_flags = random.sample(flags, 10)

if os.path.exists(DATASET_DIR) == False:
        os.makedirs(DATASET_DIR)

for flag in selected_flags:

    apply_noise_modif(flag,DATASET_DIR)
    apply_brightness_modif(flag,DATASET_DIR)
    apply_wave_modif(flag,DATASET_DIR)
