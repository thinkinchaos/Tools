root = 'D:/DATA/zhatu_coco/labelme'  # 根目录，py文件所在的路径

import os
from pathlib import Path
import json

image_names = [i.name for i in Path('D:/DATA/zhatu_coco/images').glob('*.jpg')]

for json_path in Path(root).rglob('*.json'):  # 遍历文件夹及子文件下下所有符合条件的文件
    with open(str(json_path), "r") as file:  # str是把windows路径转为string, r是只读
        dict = json.load(file)


        if dict['imagePath'] in image_names:

            for sub_dict in dict['shapes']: # 读字典指定key的value
                if sub_dict['label'] != 'soil': #判断该子字典中，指定key的value是否为所需
                    sub_dict['label'] = 'soil' # 更改

            save_path = str(json_path).replace('labelme', 'labelme_soil') #把更目录的名字改个
            save_dir_name = os.path.dirname(save_path)
            if os.path.exists(save_dir_name) is False: #如果不存在这个文件所在的新目录，则新建个
                os.makedirs(save_dir_name)

            with open(save_path, 'w') as file: #只写，写进去
                json.dump(dict, file)
