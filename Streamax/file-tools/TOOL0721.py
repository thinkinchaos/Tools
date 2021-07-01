import json
from pathlib import Path
import shutil
import os
from argparse import ArgumentParser
import xlwt


# 命令范例
# python statics.py -i abc -d 20010101

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--root", '-r', type=str, default='//192.168.133.14/shenzhen_rsycn/traffic_light/check/20200720')
    parser.add_argument("--background_new", '-b', type=str, default='//192.168.133.14/shenzhen_rsycn/traffic_light/error')
    parser.add_argument("--video_new", '-v', type=str, default='//192.168.133.14/shenzhen_rsycn/traffic_light/done_video')
    # parser.add_argument("--item", '-i', type=str, required=True)
    # parser.add_argument("--date", '-d', type=int, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    name_dirs = [i for i in Path(args.root).iterdir() if i.is_dir()]
    for name_dir in name_dirs:
        for bg_file in Path(str(name_dirs) + '/background'):
            shutil.move(str(bg_file), args.background_new)

        for vid_file in Path(str(name_dirs)).glob('*.avi'):
            shutil.move(str(vid_file), args.video_new)

    merge_path = args.root + '/merge'
    if not os.path.exists(merge_path):
        os.makedirs(merge_path)

    for all_file in Path(args.root).rglob('*.*'):
        shutil.move(str(all_file), merge_path)
