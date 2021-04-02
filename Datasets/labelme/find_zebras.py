root = '//192.168.133.14/shenzhen_rsycn/adas_parsing_data_1911_b/done'
dst_dir = '../data1'
from pathlib import Path
import shutil

json_file_set = set()
for json_file in Path(root).rglob('*.json'):
    json_file_set.add(json_file)
image_file_set = set()
for image_file in Path(root).rglob('*.jpg'):
    image_file_set.add(image_file)
# assert len(json_file_set) == len(image_file_set)
print(len(json_file_set), len(image_file_set))

zebra_json_names = set()
import json
for json_file in json_file_set:
    with open(str(json_file), "r") as f:
        dict = json.load(f)
        shapes = dict['shapes']
        for item in shapes:
            if 'zebra-crs' in item['label']:
                # print(str(json_file))
                zebra_json_names.add(json_file.name[:-5])
                shutil.copy(str(json_file), dst_dir)

for image_file in image_file_set:
    name_tmp = image_file.name[:-4]
    # print(image_file.name[:-4])
    if name_tmp in zebra_json_names:
        # print(str(image_file))
        shutil.copy(str(image_file), dst_dir)
