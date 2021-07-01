import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from pathlib import Path
    import cv2

    for p in Path('D:/DATA/COCO/train2017').glob('*.jpg'):
        img = cv2.imread(str(p))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # original_img = cv2.imread("666.png")
        img_bgr = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)

        max_idxs_of_bgr = []
        for i in range(3):
            b_hist = cv2.calcHist([img_bgr], [i], None, [256], [0, 256])

            b_hist = [j[0] for j in b_hist]
            # print(b_hist)
            max_idx_of_b = b_hist.index(max(b_hist))
            max_idxs_of_bgr.append(max_idx_of_b)

        max_channel_idx = max_idxs_of_bgr.index(max(max_idxs_of_bgr))

        if max_channel_idx == 1 and max_idxs_of_bgr[max_channel_idx]>200:
            cv2.imwrite('D:/DATA/tmp/' + p.name, img_bgr)
