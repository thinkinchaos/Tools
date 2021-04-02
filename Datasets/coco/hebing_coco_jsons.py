import json
json1='D:/DATA/guijiao/done/guidao_5_22/guidaoxian.json'
json2='D:/DATA/guijiao/done/guidao_5_22/light_crs.json'
with open(json1, 'r') as f:
    dict1=json.load(f)
with open(json2, 'r') as f:
    dict2=json.load(f)
# annos_in_dict1=dict1['annotations']
# print(len(annos_in_dict1))
for anno in dict1['annotations']:
    anno['category_id']=3
    # print(anno)

# new_annos=[]
# annos_in_dict2=dict2['annotations']
# # print(len(annos_in_dict2))
new_annos = dict2['annotations'] + dict1['annotations']
# print(len(annos_in_dict2))

new_dict={}
# print(annos_in_dict2)
# import copy
# new_dict=copy.deepcopy(dict2)
# new_dict.pop('annotations')
# new_dict.pop('categories')
# a=new_dict['annotations']
# print(dict2['categories'])
# [{'supercategory': 'none', 'id': 1, 'name': 'light'}, {'supercategory': 'none', 'id': 2, 'name': 'cross'}]
new_cat=[{'supercategory': 'none', 'id': 1, 'name': 'light'}, {'supercategory': 'none', 'id': 2, 'name': 'cross'}, {'supercategory': 'none', 'id': 3, 'name': 'guidaoxian'}]
new_img=dict1['images']
new_dict.setdefault('images', new_img)
new_dict.setdefault('annotations', new_annos)
new_dict.setdefault('categories', new_cat)

for i, anno in enumerate(new_dict['annotations']):
    anno['id']=i

# for DEMO in new_dict['images']:
#     print(DEMO)
# for DEMO in new_dict['type']:
#     print(DEMO)
# for DEMO in new_dict['annotations']:
#     if DEMO['category_id']==3:
#         print(DEMO)
for image in new_dict['categories']:
    print(image)
# images_in_dict1=[]
# for key,val in new_dict.items():
#
#     # new_dict.setdefault(key, val)
#     print(key)
#     item[]
#     print(item)
# images
# type
# annotations
# categories

# with open('D:/DATA/guijiao/done/guidao_5_22/all.json', 'w') as f:
#     json.dump(new_dict, f)