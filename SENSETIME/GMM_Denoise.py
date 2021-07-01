import os
import numpy as np
import cv2
import torch
from pathlib import Path
import matplotlib.pyplot as plt


def generate_subimages(img, s):
    masks = []
    kz = s * 2 + 1
    h0, w0, _ = img.shape
    h = h0 // kz * kz
    w = w0 // kz * kz
    img = img[:h, :w, :]
    for t in range((s * 2 + 1) * (s * 2 + 1)):
        masks.append(np.zeros(img.shape, dtype='uint8'))
    for i in range(s, h - s + 1, kz):
        for j in range(s, w - s + 1, kz):
            ss = -1 * s
            n = 0
            for p in range(ss, s + 1):
                for q in range(ss, s + 1):
                    masks[n][i + p, j + q, :] = 1
                    n += 1
    sub_imgs = []
    for mask in masks:
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # HWC to CHW
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0)
        import torch.nn.functional as F
        out = F.max_pool2d(img_tensor.float() * mask_tensor.float(), kernel_size=kz, stride=kz)
        sub_img = np.array(out.squeeze(0).permute(1, 2, 0)).astype('uint8')  # CHW to HWC
        sub_imgs.append(sub_img)
    return sub_imgs


def vis(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.title(str(img.shape))
    plt.axis('off')
    plt.pause(0.1)


def vis2(img, img2):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(121)
    plt.imshow(img)
    plt.axis('off')
    plt.title(str(img.shape))

    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    plt.subplot(122)
    plt.imshow(img2)
    plt.axis('off')
    plt.title(str(img.shape))

    plt.pause(0.1)


if __name__ == '__main__':
    path = "/home/SENSETIME/sunxin/data/front/161643347-0085-0031-161649241_off.jpg"
    noisy = cv2.imread(path)
    h0, w0, _ = noisy.shape
    s_range = [1, 2, 3, 4, 5]
    out = np.zeros(noisy.shape, dtype='float64')
    for s in s_range:
        sub_imgs = generate_subimages(noisy, s)
        avg_img = np.zeros(sub_imgs[0].shape, dtype='float64')
        mid = sub_imgs[len(sub_imgs) // 2]
        h, w, _ = mid.shape
        n = 0
        ss = -1 * s
        for p in range(ss, s + 1):
            for q in range(ss, s + 1):
                img = sub_imgs[n]
                n += 1
                avg_img += img
        avg_img /= len(sub_imgs)
        tmp = cv2.resize(avg_img, (w0, h0))
        out += tmp
    out /= len(s_range)
    out = out.astype('uint8')
    vis2(noisy, out)
    cv2.imwrite('result_' + Path(path).name, out, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
