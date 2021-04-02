from pycocotools.coco import COCO
import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path

json_file = '../../../../DATA/datasets/adas_4_3/val.json'
dataset_dir = '../../../../DATA/datasets/adas_4_3/val'
coco = COCO(json_file)
classes = ['roads', 'ground_mark', 'vehicle', 'non-motor', 'person', 'sign']


img_paths=[i for i in Path(dataset_dir).iterdir() if i.is_file()]
imgIds = list(range(1, len(img_paths)+1))
for id in imgIds:
    img_info = coco.loadImgs(id)[0]
    image = cv2.imread(os.path.join(dataset_dir, img_info['file_name']))
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype='uint8')

    for j, cls in enumerate(classes):
        label = (j + 1)*1

        catIds = coco.getCatIds(catNms=cls)
        annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds, iscrowd=0)
        anns = coco.loadAnns(annIds)

        mask_a_cls = np.zeros((h, w))
        for k in range(len(anns)):
            mask_tmp = coco.annToMask(anns[k])
            mask_a_cls += mask_tmp
        mask_a_cls = np.where(mask_a_cls > 0, label, 0).astype('uint8')

        mask += mask_a_cls

    # p=set()
    # for i in range(h):
    #     for j in range(w):
    #         # print(mask[i][j])
    #         p.add(mask[i][j])
    # print(len(p),'ss',p)
    # plt.imshow(mask)
    # plt.show()
    # break
