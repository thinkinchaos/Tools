'''
需求：1.统计日期里面的每个人的图片数量为标注量（包括background+hand_phone）；统计除开b+h的图片量为合格量；计算合格率
      2.background里的图片加文件移入error里并创建日期加姓名
      3.hand_phone里的图片加文件全部移入tmp 不用创建文件夹直接移入
      4.合并日期里面所有的图片和文件 （名字文件夹删除）并查看是否图片和文件对应 如果有不对应的创建一个文件夹放入
注意：每个名字文件夹里 有可能没有background 或者hand_phone 这种视为合格率100%
'''

import shutil
import os
from pathlib import Path

root = 'F:/t'
date = [20200723, 20200724]


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_sub_dirs_list(dir_path):
    return [i for i in Path(dir_path).iterdir() if i.is_dir()]


done_dir = root + '/done'

error_dir = root + '/error'
tmp_dir = root + '/tmp'
create_dir(error_dir)
create_dir(tmp_dir)

date_dirs = [Path(done_dir + '/' + str(i)) for i in date]
for date_dir in date_dirs:
    name_dirs = get_sub_dirs_list(date_dir)
    for name_dir in name_dirs:
        total_xml_num = len([i for i in Path(name_dir).rglob('*.xml')])
        if total_xml_num == 0:
            print('No files this dir!', name_dir)
            continue

        # 提出不匹配的文件
        qualified_num = 0
        qualified_file_names = [i.name for i in Path(name_dir).glob('*.*')]
        for qualified_file in Path(name_dir).glob('*.*'):
            if qualified_file.name[:-3] + 'jpg' in qualified_file_names and qualified_file.name[:-3] + 'xml' in qualified_file_names:
                qualified_num += 1
            else:
                unmatched_dir = root + '/unmatched'
                create_dir(unmatched_dir)
                try:
                    shutil.move(str(qualified_file), unmatched_dir)
                except shutil.Error:
                    os.remove(str(qualified_file))

        # 统计合格数和合格率
        assert qualified_num % 2 == 0
        qualified_num //= 2
        qualified_rate = round((float(qualified_num) / total_xml_num * 100), 2)
        print('Date:{}, Name:{}, Qualified num:{}, Qualified rate:{}%'.format(date_dir.name, name_dir.name, qualified_num, qualified_rate))

        # 提出合格的文件
        sub_dirs = get_sub_dirs_list(name_dir)
        for file in Path(name_dir).glob('*.*'):
            try:
                shutil.move(str(file), str(date_dir))
            except shutil.Error:
                os.remove(str(file))

        # 提出background和hand_phone的文件
        for sub_dir in sub_dirs:
            transfer_dir = ''
            if sub_dir.name == 'background':
                transfer_dir = error_dir + '/' + date_dir.name + '/' + name_dir.name
            elif sub_dir.name == 'hand_phone':
                transfer_dir = tmp_dir + '/' + date_dir.name + '/' + name_dir.name
            else:
                print('Unexpected sub dir!', sub_dir)
                continue
            create_dir(transfer_dir)
            for file in Path(sub_dir).glob('*.*'):
                try:
                    shutil.move(str(file), transfer_dir)
                except shutil.Error:
                    os.remove(str(file))

            shutil.rmtree(str(sub_dir))
        shutil.rmtree(str(name_dir))
