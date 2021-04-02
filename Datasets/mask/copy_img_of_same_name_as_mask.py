from pathlib import Path
import shutil
import cv2
import os
import random
from PIL import Image

mask_names = [i.name for i in Path('D:/DATA/zhatu/mask').glob('*.png')]
img_names = [i.name for i in Path('D:/DATA/zhatu/image').glob('*.jpg')]

# for i in mask_names:
#     img_name = i[:-4]+'.jpg'
#     if img_name in img_names:
#         print(img_name)
#         shutil.copy('D:/DATA/zhatu0618/'+img_name, 'D:/DATA/zhatu/image')

random.shuffle(mask_names)
t = mask_names[:100]
for i in t:
    shutil.copy('D:/DATA/zhatu/mask/' + i, 'D:/DATA/zhatu/test/mask')
    shutil.copy('D:/DATA/zhatu/image/' + i[:-3]+'jpg', 'D:/DATA/zhatu/test/image')