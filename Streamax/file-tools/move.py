from pathlib import Path
import shutil
import os
import argparse
import sys
import copy

# 在该py文件所在位置打开cmd, 输入如下格式的指令。注意，应使用右斜杠。可输入多个名字，空格隔开。
# python move.py --root //192.168.133.14/shenzhen_rsycn/BSD_drivable_area_car_people/nodone --names luoao1-luoao2 --num 2000 --out //192.168.133.14/shenzhen_rsycn/out

parser = argparse.ArgumentParser()

# 根目录，支持4种情况，移动足够的文件数为止。
# 优先移动文件夹中的文件，优点移动jpg，不移动没有jpg对应的json或xml。
parser.add_argument("--root", type=str, default='E:/t')

# 要存的名字。自动在输出目录下新建该文件夹。
parser.add_argument("--names", type=str, default='p1-p2')

# 要移动的数量。
parser.add_argument("--num", type=int, default=50)

# 移动到的路径。如果不指定，或者指定的路径不存在，则采用默认的输出路径root。
parser.add_argument("--out", type=str, default='')

args = parser.parse_args()


def mkdir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def move_files(moved_num, file_dir, dst_dir, thresh):
    jpgs = [i for i in Path(file_dir).glob('*.jpg')]

    dst_paths = []
    for i, jpg in enumerate(jpgs):
        if len(dst_paths) < thresh:
            dst_paths.append(jpg)

        json = str(jpg).replace('jpg', 'json')
        if os.path.exists(json) and len(dst_paths) < thresh:
            dst_paths.append(json)

        xml = str(jpg).replace('jpg', 'xml')
        if os.path.exists(xml) and len(dst_paths) < thresh:
            dst_paths.append(xml)

        if len(dst_paths) + moved_num == thresh:
            break

    if len(dst_paths) > 0:
        mkdir(dst_dir)
        for idx, path in enumerate(dst_paths):
            print('{}\t\t{}\t\t{}'.format(idx + 1 + moved_num, path, Path(dst_dir)))
            shutil.move(str(path), dst_dir)
        return len(dst_paths) + moved_num
    else:
        return moved_num


if __name__ == '__main__':
    root = args.root
    names = args.names.split('-')
    thresh = args.num
    if not os.path.exists(args.out):
        out = str(Path(root).parent)
    else:
        out = args.out
    print(vars(args))


    for name in names:
        print('\n#### Idx #### Moving_file #### Save_name ####')
        dirs = [i for i in Path(root).iterdir() if i.is_dir()]

        flag = True
        moved_num = 0
        while flag:
            for sub_dir in dirs:
                moved_num_tmp = move_files(moved_num, sub_dir, out + '/' + name + '/' + sub_dir.name, thresh=args.num)
                moved_num += moved_num_tmp
                if moved_num >= thresh:
                    flag = False
                    break

            if flag:
                moved_num = move_files(moved_num, root, out + '/' + name, thresh=args.num)
                if moved_num >= thresh:
                    flag = False
                    break

            if moved_num == 0:
                break
