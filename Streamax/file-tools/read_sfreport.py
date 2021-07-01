# encoding=utf-8
from pathlib import Path
import os

info = []
dirs = [i for i in Path(u'//192.168.133.14/share/实习生日报').iterdir() if i.is_dir()]
for ddd in dirs:
    sub_dirs = [i for i in Path(ddd).iterdir() if i.is_dir()]
    for d in sub_dirs:
        dd = [i for i in os.listdir(str(d)) if '孙鑫' in i]
        if len(dd) > 0:
            path = str(d) + '/' + dd[0]
            with open(path, 'r', encoding='utf-8') as f:
                lines = [i.strip() for i in f.readlines() if '。' in i]
                info = info + lines
for i in info:
    print(i)
