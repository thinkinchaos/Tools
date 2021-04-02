from pathlib import Path
import shutil
import cv2
img = cv2.imread('D:/0000000000000000-200604-145311-145811-0000012105400000001999.jpg')
for path in Path('D:/val2017').glob('*.jpg'):
    cv2.imwrite(str(path).replace('val2017', 'v'), img)