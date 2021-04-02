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
    parser.add_argument("--root", '-r', type=str, default='//192.168.133.14/share')  # 根据需求把share改为shenzhen_rsycn
    parser.add_argument("--item", '-i', type=str, required=True)
    parser.add_argument("--date", '-d', type=int, required=True)
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    item = args.item
    date = str(args.date)
    root = args.root + '/' + item + '/done/' + date

    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet(date, cell_overwrite_ok=True)
    sheet.write(0, 0, 'person')
    sheet.write(0, 1, 'total_num')
    sheet.write(0, 2, 'ok_num')
    sheet.write(0, 3, 'error_num')
    sheet.write(0, 4, 'error_rate')
    sheet.write(0, 5, 'ok_rate')

    dirs = [i for i in Path(root).iterdir() if i.is_dir()]

    for i, dir in enumerate(dirs):

        total = 0
        positive = 0
        negative = 0

        bk_path = os.path.join(str(dir), 'background')
        if os.path.exists(bk_path):
            num_bk = len([i for i in Path(bk_path).glob('*.json')])
            total += num_bk
            negative += num_bk
            new_bk_path = str(dir).replace('done', 'error')
            if not os.path.exists(new_bk_path):
                os.makedirs(new_bk_path)
            for bk_file in Path(bk_path).glob('*.*'):
                shutil.move(str(bk_file), os.path.join(new_bk_path, bk_file.name))

        for json_path in Path(dir).glob('*.json'):
            total += 1
            with open(str(json_path), 'r') as f:
                dict = json.load(f)
            if 'flags' in dict:
                if 'bdcheckedOk' in dict['flags']:
                    flag = dict['flags']['bdcheckedOk']
                    if flag == 1:
                        positive += 1
                    elif flag == 0:
                        negative += 1
                else:
                    positive += 1
            else:
                positive += 1

        if total > 0:
            n_rate = negative / float(total)
            p_rate = positive / float(total)
            n_rate = round(n_rate, 4)
            p_rate = round(p_rate, 4)

            sheet.write(i + 1, 0, dir.name)
            sheet.write(i + 1, 1, total)
            sheet.write(i + 1, 2, positive)
            sheet.write(i + 1, 3, negative)
            sheet.write(i + 1, 4, n_rate)
            sheet.write(i + 1, 5, p_rate)
        else:
            sheet.write(i + 1, 0, 0)
            sheet.write(i + 1, 1, 0)
            sheet.write(i + 1, 2, 0)
            sheet.write(i + 1, 3, 0)
            sheet.write(i + 1, 4, 0)
            sheet.write(i + 1, 5, 0)

    for file in Path(root).rglob('*.*'):
        if 'background' not in str(file) and 'negatives' not in str(file):
            try:
                shutil.move(str(file), str(root))
            except shutil.Error:
                os.remove(str(file))

    for dir in dirs:
        shutil.rmtree(str(dir))

    book.save(args.root + '/' + item + '/done/' + date + '.xls')
