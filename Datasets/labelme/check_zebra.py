# path='D:/DATA/zebra_5_9_coco'
# import json
# with open(path+'/train.json', 'r') as f:
#     dict = json.load(f)
# # print(dict)
# # print(len(dict), dict)
# seg=dict['annotations']
# img=dict['images']
# cat=dict['categories']
# # print(seg, img, cat)
# for i in seg:
#     c = i['category_id']
#     if c!=1:
#         print(i)
# #{'segmentation': [[1654, 381, 1658, 353, 1658, 353]], 'iscrowd': 0, 'image_id': 227, 'bbox': [1654.0, 353.0, 4.0, 28.0], 'area': 112.0, 'category_id': 2, 'id': 2281}
#
# for j in img:
#     id = j['id']
#     if id == 227:
#         print(j)
#     name = j['file_name']
#     if name == '0000000000000000-190501-092526-092534-000008000140.2640000000131_00(1).jpg':
#         print('ss', j)
# #{'height': 396, 'width': 1732, 'id': 227, 'file_name': '0000000000000000-190427-184016-184020-000008247490.2640000000060_00.jpg'}
#
# for k in cat:
#     print(k)
# # print(type(seg))
#
# path='D:/DATA/zebra_5_9_coco/json'
from pathlib import Path
import json
# for path in Path(path).glob('*.json'):
#     with open(str(path), 'r') as f:
#         dic = json.load(f)
#     img = dic['imagePath']
#     if '0000000000000000-190501-092526-092534-000008000140.2640000000131_00(1).jpg' == img:
#         print(path)
#     # print(img)
#     # for k in dic:
#     #     print(k)
#     # break
import shutil
image_path='D:/DATA/zebra_5_6_labelme/image'
images=[i.name for i in Path(image_path).glob('*.jpg')]
json_path='D:/DATA/zebra_5_9_coco/json'
for path in Path(json_path).glob('*.json'):
    with open(str(path), 'r') as f:
        dic = json.load(f)
    img = dic['imagePath']
    if img in images:
        pass
    else:
        shutil.move(str(path), 'D:/DATA/zebra_5_6_labelme')
        # print(img)
        print(path)
    # if '0000000000000000-190501-092526-092534-000008000140.2640000000131_00(1).jpg' == img:
    #     print(path)
    # print(img)
    # for k in dic:
    #     print(k)
    # break


