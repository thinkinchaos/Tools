from pycocotools.coco import COCO

annFile="d:/mydata/adas_parsing_data_1907/coco_v1910_train.json"
annFile2="d:/mydata/adas_parsing_data_1907/1.json"

coco = COCO(annFile)
cat_num = len(coco.dataset['categories'])

img_num = len(coco.dataset['images'])

ann_num = len(coco.dataset['annotations'])

print(cat_num, img_num, ann_num)
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))


