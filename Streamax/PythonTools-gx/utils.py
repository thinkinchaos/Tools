#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/22 11:09
# @Author  : Sillet

import os
import sys
import shutil


def view_bar(num, total):
    rate = num / total
    rate_num = int(rate * 100)
    # r = '\r %d%%' %(rate_num)
    r = '\r%s>%d%%\t%d' % ('=' * rate_num, rate_num, num)
    sys.stdout.write(r)
    sys.stdout.flush


def copyFilesFromList(srcFileList, dstPath):
    if not os.path.isdir(dstPath):
        print('%s is not dir!')
        return

    size = len(srcFileList)
    for i, file in enumerate(srcFileList):
        if os.path.exists(file):
            shutil.copy(file, dstPath)
        view_bar(i, size)

    print('copy done!')