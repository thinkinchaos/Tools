from PIL import Image
from pathlib import Path
import numpy as np
import os
hard_img_dir = 'D:/DATA/Greens2'
save_mask_dir = 'D:/DATA/labels'
p_mask = Path(save_mask_dir)
if not os.path.exists(save_mask_dir):
    os.makedirs(save_mask_dir)

for image_path in Path(hard_img_dir).glob('*.jpg'):
    name = image_path.name
    image = Image.open(str(image_path))
    size = image.size
    mask_np = np.zeros((size[1], size[0]), dtype='uint8')
    mask = Image.fromarray(mask_np.astype('uint8')).convert('L')
    mask.save(os.path.join(save_mask_dir, name[:-3] + 'png'))
    assert image.size == mask.size
