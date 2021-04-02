import cv2

# import os
root = 'C:/Users/Walter/Desktop/infrared'
from pathlib import Path


def img2video(imgpath, video):
    img_list = [str(i) for i in Path(imgpath).iterdir() if i.is_file()]
    # img_list = os.listdir(imgpath)
    # img_list.sort(key=lambda t: int(t[:t.index('.')]))

    video_w, video_h = 700, 520
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vw = cv2.VideoWriter(video, fourcc, 12, (video_w, video_h))

    for img in img_list:
        frame = cv2.imread(img)  # any depth, cus default 8bit
        # frame = frame[20:540, 10:710, :]
        # # print(frame.shape)
        cv2.imshow('ss', frame)
        cv2.waitKey()
        # vw.write(frame)
        # print('write', img)

    vw.release()


if __name__ == '__main__':
    img2video(imgpath=root, video='E:/SWUST/Paper/EXP/demo.avi')
