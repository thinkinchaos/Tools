from pathlib import Path
import shutil
xmls = [i.name for i in Path('D:/sunxin2/Annotations').rglob('*.xml')]
for jpg in Path('D:/result').rglob('*.jpg'):
    if jpg.name[:-3]+'xml' in xmls:
        # print('err')
        shutil.copy(str(jpg), 'D:/sunxin2/JPEGImages')

# jpgs = [i.name for i in Path('D:/12955').rglob('*.jpg')]
# for xml in Path('D:/12955').rglob('*.xml'):
#     if xml.name[:-3]+'jpg' in jpgs:
#         shutil.copy(str(xml), 'D:/12955')
import cv2
from pathlib import Path
import shutil
import os
from tqdm import tqdm

names = [
    'liang_zhiqiang',
    'pang_zhongxiang',
    'fan_xinchuan',
    'chen_guodong',
    'liu_tianci',
    'deng_lei',

    'yao_chao',
    'li_jin',
    'huang_xiaoli',
    'nie_yu',
    'chen_xulei',
]
num_person = len(names)
paths = [i for i in Path('../tiny').glob('*.jpg')]
num_data_each_person = len(paths) // num_person
save_dir = '../dataset'
for i, name in enumerate(names):
    save_dir_tmp = save_dir + '/' + name
    if not os.path.exists(save_dir_tmp):
        os.makedirs(save_dir_tmp)
    start_id = i * num_data_each_person
    paths_tmp = paths[start_id:start_id + num_data_each_person]
    for path_tmp in paths_tmp:
        shutil.copy(str(path_tmp), save_dir_tmp)
    print(name, '\tfinish')