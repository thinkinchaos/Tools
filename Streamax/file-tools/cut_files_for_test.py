from PIL import Image
from pathlib import Path
import numpy as np
import os
import shutil

test_image_dir = 'D:/DATA/zebra_5_6_hard_neg/test-env/DEMO'

src_masks_dir = 'D:/DATA/zebra_5_6_hard_neg/mask'
save_masks_dir = 'D:/DATA/zebra_5_6_hard_neg/test-env/mask'

for glob_mask in Path(src_masks_dir).glob('*.png'):
    for test_image in Path(test_image_dir).glob('*.jpg'):
        if test_image.name[:-3] == glob_mask.name[:-3]:
            shutil.move(str(glob_mask), save_masks_dir)
