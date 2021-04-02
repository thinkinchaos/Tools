import os
import os.path
import sys
import cv2 as cv
import random
import shutil
from SmokingTrainSampleChoose import *

REZISE_W = 128
REZISE_H = 128

srcFile = 'G:\Train_Data\DataSet\Label_DSM\Annotation_1711'
dstFile = 'G:\Train_Data\DataSet\Label_DSM/tooth_1104'
fullFile = 'G:\Train_Data\DataSet\Label_DSM\smoking_full_9.19'

landMarkFolder = 'Q:/F_yghan/DSM/samples' \
                 '/face_alignment/rm/0/streamax/new/bus_car'
smokeNegFolder = 'G:/Train_Data/DataSet/Label_DSM/smoking_negative_landmark'


def view_bar(num, total):
    rate = num / total
    rate_num = int(rate * 100)
    # r = '\r %d%%' %(rate_num)
    r = '\r%s>%d%%\t%d' % ('=' * rate_num, rate_num, num)
    sys.stdout.write(r)
    sys.stdout.flush


def VOC2Cliper(xml_folder):
    exePath = 'vocxml2clip.exe '
    parameter = '-f ' + xml_folder + ' -d ' + xml_folder + ' -F .'
    print(exePath + parameter)
    if (os.system(exePath + parameter) == 0):
        print('Done!')


def saveImageFromCliper(path, labelTXT, saveFullimgPath=None, non=False):
    print('saveImageFromCliper process %s:' % path)
    if not os.path.exists(os.path.join(path, labelTXT)):
        return
    with open(os.path.join(path, labelTXT)) as labelFile:
        lines = labelFile.readlines()
        sum_numbel = len(lines)
        for i, line in enumerate(lines):
            dataList = line.split(' ')
            imgName = dataList[0]
            imgNum, x, y, w, h = [int(i) for i in dataList[1:6]]
            imgPath = os.path.join(path, imgName[2:])
            img = cv.imread(imgPath)
            if img is None:
                continue
            if not saveFullimgPath is None:
                if not os.path.isdir(saveFullimgPath):
                    os.makedirs(saveFullimgPath)
                cv.imwrite(os.path.join(saveFullimgPath, imgName), img)
            smokeImg = img[y:(y + h), x:(x + w)]
            cv.imwrite(os.path.join(dstFile, imgName), smokeImg)

            if (non):
                cv.rectangle(img, (x, y), (x + w, y + h), 255, -1)
                # TODO:save img
            view_bar(i, sum_numbel)
        print('Done!')


def smokePick(srcFile, dstFile, saveFullPath=None):
    if not os.path.isdir(srcFile):
        os.makedirs(srcFile)
    if not os.path.isdir(dstFile):
        os.makedirs(dstFile)
    for folder in os.listdir(srcFile):
        folderPath = os.path.join(srcFile, folder)
        VOC2Cliper(folderPath)
        saveImageFromCliper(folderPath, 'tooth.txt', saveFullimgPath=saveFullPath)
    if not os.path.isdir(dstFile + '_vis'):
        os.makedirs(dstFile + '_vis')
    command = 'D:\PICO_train\CreateTrainData.exe -i ' \
              + dstFile + ' -o ' \
              + dstFile + '_vis -vis LBPH -th 0.98 -hp false'
    os.system(command)


def getSmokeRect(ptsFile):
    pointList = ReadPTS(ptsFile)
    return SomkingDetectRegion(pointList)


def saveSmokeNegativeImgfromLandMark():
    folderList = os.listdir(landMarkFolder)
    for folder in folderList:
        folderPath = os.path.join(landMarkFolder, folder)
        dvrList = os.listdir(folderPath)
        for dvrFolder in dvrList:
            dvrPath = os.path.join(folderPath, dvrFolder+'/landmark')
            if not os.path.isdir(dvrPath):
                continue
            fileList = list(set([i.split('.')[0] for i in os.listdir(dvrPath)]))
            for file in fileList:
                filePath = os.path.join(dvrPath, file)
                imgPath = filePath + '.jpg'
                ptsPath = filePath + '.pts'
                smokeRect = getSmokeRect(ptsPath)
                img = cv.imread(imgPath)
                if img is None:
                    print(('%s is open failed!') % (imgPath))
                    continue
                smokeImg = img[smokeRect.y:(smokeRect.y + smokeRect.high), smokeRect.x:(smokeRect.x + smokeRect.width)]
                writeFile = os.path.join(smokeNegFolder, file) + '.jpg'
                cv.imwrite(writeFile, smokeImg)
            print(dvrPath + ' Done!')


def saveSmokeRandomBackground(srcFolderPath, dstFolderPath):
    folderList = os.listdir(srcFolderPath)
    for folder in folderList:
        folderPath = os.path.join(srcFolderPath, folder)
        fileList = os.listdir(folderPath)
        for file in fileList:
            imgPath = os.path.join(folderPath, file)
            img = cv.imread(imgPath)
            if img is None:
                print(('%s is open failed!') % (imgPath))
                continue
            imgW = img.shape[1]
            imgH = img.shape[0]
            rectW = random.randint(128, 256)
            rectH = random.randint(128, 256)
            rectX = random.randint(1, imgW - rectW)
            rectY = random.randint(1, imgH - rectH)
            dstImg = img[rectY:(rectY + rectH), rectX:(rectX + rectW)]
            dstImg = cv.resize(dstImg, (128, 128))
            cv.imwrite(os.path.join(dstFolderPath, file), dstImg)


def randomData(srcFolder):
    fileList = os.listdir(srcFolder)
    random.shuffle(fileList)
    filePos = int(len(fileList) * 0.8)
    for i, file in enumerate(fileList):
        filePath = os.path.join(srcFolder, file)
        dstFolder = srcFolder + '_train'
        if i > filePos:
            dstFolder = srcFolder + '_test'
        img = cv.imread(filePath)
        if img is None:
            print(('%s is open failed!') % (filePath))
            continue
        dstFilePath = os.path.join(dstFolder, file)
        cv.imwrite(dstFilePath, img)
        print(file)
    print('Done!')


if __name__ == '__main__':
    smokePick(srcFile, dstFile)
    # saveSmokeNegativeImgfromLandMark()
# saveSmokeRandomBackground('Q:\F_public\public\DSM_VIDEO/background_indoor/background_indoor',
#                           'G:\Train_Data/DataSet/Label_DSM/smoking_sample/background')
# randomData('G:\\Train_Data\\DataSet\\Label_DSM\\smoking_sample\\positive')
