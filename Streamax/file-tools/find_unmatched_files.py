from pathlib import Path

root = ''  # 文件夹的路径，把\全部替换为/
json_names = [i.name[:-5] for i in Path(root).glob('*.json')]
img_names = [i.name[:-4] for i in Path(root).glob('*.jpg')]
for json_name in json_names:
    if json_name not in img_names:
        print(json_name + '.json')

for img_name in img_names:
    if img_name not in json_names:
        print(img_name + '.jpg')