root = 'D:/DATA/guijiao/done/guidao'  # 根目录，py文件所在的路径

import os
from pathlib import Path
import json

for json_path in Path(root).rglob('*.json'):  # 遍历文件夹及子文件下下所有符合条件的文件
    with open(str(json_path), "r") as file:  # str是把windows路径转为string, r是只读
        dict = json.load(file)

        for sub_dict in dict['shapes']: # 读字典指定key的value

            if sub_dict['label'] != 'guidaoxian': #判断该子字典中，指定key的value是否为所需
                print(json_path.name)

