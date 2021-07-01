from pathlib import Path
import random
import shutil


def mkdir(dir):
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)


dir = '//192.168.133.15/workspace/sunxin/DATA/adas/road0717'
# mkdir(dir)
mask_names = [i.name for i in Path(dir + '/labels').glob('*.png')]
random.shuffle(mask_names)
test_mask_names = mask_names[:int(len(mask_names) * 0.1)]
test_img = dir + '/test/images'
test_mask = dir + '/test/labels'
mkdir(test_img)
mkdir(test_mask)
for i in test_mask_names:
    print(i, i[:-3] + 'jpg')
    shutil.move(dir + '/labels/' + i, test_mask)
    shutil.move(dir + '/images/' + i[:-3] + 'jpg', test_img)
