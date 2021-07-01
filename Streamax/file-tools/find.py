root = 'E:/image'
save = 'E:/ggg'

import os
names = os.listdir(root)
if not os.path.exists(save):
    os.makedirs(save)
f = open("n.txt","r")
lines = f.readlines()      #读取全部内容 ，并以列表方式返回
import shutil
for line in lines:
    # print(line)
    line = line.strip()
    if line in names:
        print(line)
        shutil.copy(root + '/' + line, save)