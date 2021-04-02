from pathlib import Path
import shutil
import cv2
import os
import random
from PIL import Image


def create_bw_mask():
    src_img = 'D:/zebra_crs_to_0424/DEMO'
    src_mask = 'D:/zebra_crs_to_0424/mask'

    dst_img = 'D:/dst/DEMO'
    dst_mask = 'D:/dst/mask'
    os.makedirs(dst_img)
    os.makedirs(dst_mask)

    for mask_path in Path(src_mask).glob('*.jpg'):
        mask_name = mask_path.name
        mask = cv2.imread(str(mask_path))
        img = cv2.imread(src_img + '/' + mask_name)

        mask = cv2.resize(mask, (1000, 500), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (1000, 500))

        assert mask.shape == img.shape
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

        cv2.imwrite(dst_mask + '/' + mask_name, mask)
        cv2.imwrite(dst_img + '/' + mask_name, img)


def divide_data():
    # src_img = 'D:/dst_img/DEMO'
    src_mask = 'D:/dst/mask'

    train = 'D:/dst/train'
    trainannot = 'D:/dst/trainannot'
    os.makedirs(train)
    os.makedirs(trainannot)
    val = 'D:/dst/val'
    valannot = 'D:/dst/valannot'
    os.makedirs(val)
    os.makedirs(valannot)
    test = 'D:/dst/test-env'
    testannot = 'D:/dst/testannot'
    os.makedirs(test)
    os.makedirs(testannot)

    mask_paths = [i for i in Path(src_mask).iterdir() if i.is_file()]
    random.shuffle(mask_paths)
    for i, mask_path in enumerate(mask_paths):
        if i < 10:
            shutil.copy(str(mask_path), testannot)
            shutil.move(str(mask_path).replace('mask', 'DEMO'), test)
        elif i >= 10 and i < 100:
            shutil.copy(str(mask_path), valannot)
            shutil.move(str(mask_path).replace('mask', 'DEMO'), val)
        else:
            shutil.copy(str(mask_path), trainannot)
            shutil.move(str(mask_path).replace('mask', 'DEMO'), train)

def create_pil_png_mask():
    src_img = 'D:/zebra_crs_to_0424/DEMO'
    src_mask = 'D:/zebra_crs_to_0424/mask'

    dst_img = 'D:/zebra_camvid_pil/train'
    dst_mask = 'D:/zebra_camvid_pil/trainannot'
    os.makedirs(dst_img)
    os.makedirs(dst_mask)

    for mask_path in Path(src_mask).glob('*.jpg'):
        mask_name = mask_path.name
        mask = Image.open(str(mask_path)).convert('RGB')
        img = Image.open(src_img + '/' + mask_name).convert('RGB')

        img.save(dst_img + '/' + mask_name)

        width = mask.size[0]
        height = mask.size[1]
        for h in range(0, height):
            for w in range(0, width):
                pixel = mask.getpixel((w, h))
                print(pixel)
        break
                # if pixel != 0 and pixel != 255:
                #     print(pixel, ' ')
                # if pixel > 50:
                #     label.putpixel((w, h), 255)
                # else:
                #     label.putpixel((w, h), 0)


if __name__ == "__main__":
    # create_bw_mask()
    # divide_data()
    create_pil_png_mask()


    # from PIL import Image
    # img = Image.open('C:\\Users\\Gigabyte\\Desktop\\1.png')
    # width = img.size[0]
    # height = img.size[1]
    # for h in range(0, height):
    #     for w in range(0, width):
    #         pixel = img.getpixel((w, h))
    #         print(pixel)


