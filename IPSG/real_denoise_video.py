import cv2
from pathlib import Path
import shutil
import os
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import math


def PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def change_cv2_draw(image, strs, local, colour, sizes):
    cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("SIMYOU.TTF", sizes, encoding="utf-8")
    draw.text(local, strs, colour, font=font)
    image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return image


vw = cv2.VideoWriter('denoise_show.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1460, 580))

vid1 = cv2.VideoCapture('demo.avi')
vid2 = cv2.VideoCapture('demo_denoised.avi')
n1 = int(vid1.get(cv2.CAP_PROP_FRAME_COUNT))
n2 = int(vid2.get(cv2.CAP_PROP_FRAME_COUNT))
assert n1 == n2
for i in range(n1 - 1):
    vid1.set(cv2.CAP_PROP_POS_FRAMES, i)
    vid2.set(cv2.CAP_PROP_POS_FRAMES, i)
    _, f1 = vid1.read()
    _, f2 = vid2.read()
    dp = PSNR(f1, f2)

    f1 = cv2.copyMakeBorder(f1, 40, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    f2 = cv2.copyMakeBorder(f2, 40, 20, 0, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    f1 = change_cv2_draw(f1, '降噪前', (20, 0), (255, 0, 0), 30)
    f2 = change_cv2_draw(f2, '降噪后', (0, 0), (255, 0, 0), 30)
    f2 = change_cv2_draw(f2, 'PSNR ' + str(round(dp, 2)), (550, 0), (0, 255, 0), 30)

    f3 = cv2.hconcat([f1, f2])
    # cv2.imshow('s', f3)
    # cv2.waitKey()
    # print(f3.shape)
    vw.write(f3)
vw.release()
