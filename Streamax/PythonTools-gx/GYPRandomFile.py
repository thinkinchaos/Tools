import os
import os.path
import random

filePath = "./train.txt"
dstPath = "./randomTrain.txt"

folderList = []
dataDict = {}
retList = []


def randomFileFromFolder(folder, num):
    if num > len(dataDict[folder]):
        return
    fileList = random.sample(dataDict[folder], num)
    for file in fileList:
        dataDict[folder].remove(file)
    fileList = [os.path.join(folder, i) for i in fileList]
    if len(dataDict[folder]) < 2:
        folderList.remove(folder)
    retList.extend(fileList)


def _main():
    sumNum = 0
    with open(filePath) as fp:
        for line in fp:
            imgPath, imgNum = line.split('/')[1:]
            if imgPath not in folderList:
                folderList.append(imgPath)
                dataDict[imgPath] = []
            dataDict[imgPath].append(imgNum)
            sumNum += 1
            if sumNum % 1000 == 0:
                print('loading:%d' % sumNum)

    countNum = 0
    while len(folderList) > 2:
        countNum += 1
        if countNum % 1000 == 0:
            print("random:%d" % countNum)
        randomfolder = random.sample(folderList, 3)
        file_control = True
        for i, folder in enumerate(randomfolder):
            if len(dataDict[folder]) > 1:
                tmp = randomfolder[0]
                randomfolder[0] = randomfolder[i]
                randomfolder[i] = tmp
                file_control = False
                break
        if file_control:
            continue
        for i, folder in enumerate(randomfolder):
            if i == 0:
                randomFileFromFolder(folder, 2)
            else:
                randomFileFromFolder(folder, 1)

    with open(dstPath, 'w') as wp:
        wp.writelines(retList)


if __name__ == '__main__':
    _main()
    print('Done!')
