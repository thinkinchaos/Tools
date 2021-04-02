import cv2
from pathlib import Path

root = r'C:\Users\Walter\Desktop\infrared'
save_dir = 'D:/img/'
for i in Path(root).glob('*.avi'):
    cap = cv2.VideoCapture(str(i))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # dstSize = (int(frameWidth / 2), int(frameHeight / 2))
    total_frame_num_VIDEO = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # out = cv2.VideoWriter("../../../../DATA/re.mp4", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, dstSize)
    # print(total_frame_num_VIDEO)
    for frame_idx in range(total_frame_num_VIDEO - 1):
        ret, frame = cap.read()
        img = frame[114:450, 8:-40]
        # print(340//16*16, 600//16*16)
        # print(img.shape)
        # img = cv2.resize(frame, dstSize)
        # file = '//192.168.133.14/share/' + str(frame_idx).zfill(5) + '.jpg'

        img = cv2.flip(img, 0)
        save_img_name = save_dir + i.name + '.' + str(frame_idx) + '.png'
        cv2.imwrite(save_img_name, img)

