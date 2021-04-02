from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2
dataset_dir = '//192.168.133.15/workspace/sunxin/DATA/zebra_5_20'
for mask_path in Path('C:/panopticapi-master/converted_data/adas_seg_val').glob('*.png'):
    # img_path=dataset_dir+'/DEMO/'+mask_path.name[:-3]+'jpg'
    # img=cv2.imread(img_path)
    # mask = cv2.imread()
    mask = Image.open(str(mask_path))
    # mask=cv2.imread(str(mask_path))

    # np.where(mask>1, 255, 0)
    # np.where(mask == 0, 0, 0)
    # np.where(mask == 2, , 0)
    #
    # img=Image.fromarray(mask)
    # # img = Image.open()
    plt.imshow(mask)
    plt.show()
    print(len(mask.split()))