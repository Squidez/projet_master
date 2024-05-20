import os
import random
from PIL import Image

from noise_modif import apply_noise_modif
from wave_modif import apply_wave_modif
from brightness_modif import apply_brightness_modif

FLAG_LIST_DIR = 'flag_list.txt'
DATASET_DIR = 'input_data'

with open(FLAG_LIST_DIR, 'r') as f:
    flags = [line[:-1] for line in f]

selected_flags = random.sample(flags, 10)

if os.path.exists(DATASET_DIR) == False:
        os.makedirs(DATASET_DIR)

for flag in selected_flags:

    flag_dir = '{0}/{1}'.format(DATASET_DIR,flag)

    if os.path.exists(flag_dir) == False:
        os.makedirs(flag_dir)

    apply_noise_modif(flag,flag_dir)
    apply_brightness_modif(flag,flag_dir)
    apply_wave_modif(flag,flag_dir)
