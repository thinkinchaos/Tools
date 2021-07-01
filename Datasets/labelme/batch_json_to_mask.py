import random
import shutil
from PIL import Image
import numpy as np
import os
from pathlib import Path
import cv2
import json
import os
# import warnings
from PIL import Image
# import yaml
from labelme import utils
import base64
from pathlib import Path


def labelme_jsons_to_masks():
    json_dir = '//192.168.133.14/shenzhen_rsycn/zhatu_paosa/done/V1.1'
    image_dir = '//192.168.133.14/shenzhen_rsycn/zhatu_paosa/done/image'
    out_dir = 'D:/zhatu0814'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    json_paths = [i for i in Path(json_dir).rglob('*.json')]
    print(len(json_paths))
    for json_path in json_paths:
        file_name = json_path.name[:-5]
        data = json.load(open(str(json_path)))
        try:
            imageData = data.get('imageData')
            if not imageData:
                image_path = os.path.join(image_dir, data['imagePath'])
                with open(image_path, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')
                img = utils.img_b64_to_arr(imageData)
                lbl, lbl_names = utils.shape.labelme_shapes_to_label(img.shape, data['shapes'])
                Image.fromarray(lbl).save(os.path.join(out_dir, '{}.png'.format(file_name)))
        except OSError:
            # print(json_path)
            pass
        continue


if __name__ == '__main__':
    # del_tmp_files('./Results')
    # merge_images_into_video()
    labelme_jsons_to_masks()
