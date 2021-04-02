import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import xml.etree.ElementTree as ET

##测试图片及保存路径
dataset_path = './dashboardImage'
dataset_new_path = './New_dataset'


#########原始图片及标签迁移
def origi_image(ima_name , img_path , xml_path , save=False):
    
    ima_data = cv.imread(img_path)
    
    tree = ET.parse(xml_path)
    root = tree.getroot()

    #更新TREE
    root.find("filename").text = ima_name + "_origi" + ".jpg"
    root.find("path").text = dataset_new_path + "/" + ima_name + "_origi" + ".jpg"

    if save:
        cv.imwrite(dataset_new_path + "/" + ima_name + "_origi" + ".jpg" , ima_data)
        tree.write(dataset_new_path + "/" + ima_name + "_origi" + ".xml")


#########缩放图片
def resize_image(ima_name , img_path , xml_path , newsize , save=False):
    
    ima_data = cv.imread(img_path)
    tree = ET.parse(xml_path)
    root = tree.getroot() 

    #更新image
    ima_data = cv.resize(ima_data , (newsize,newsize))
    
    
    #更新size
    size = root.find('size')
    size.find('width').text =str(newsize)
    size.find('height').text =str(newsize)
    size.find('depth').text =str(3)

    #更新BOX
    for box in root.iter('bndbox'):
        
        box.find('xmin').text = str(int(float(box.find('xmin').text)*(newsize/512)))
        box.find('ymin').text = str(int(float(box.find('ymin').text)*(newsize/512)))
        box.find('xmax').text = str(int(float(box.find('xmax').text)*(newsize/512)))
        box.find('ymax').text = str(int(float(box.find('ymax').text)*(newsize/512)))

    if save:
        cv.imwrite(dataset_new_path + "/" + ima_name + "_resiz" + ".jpg" , ima_data)
        tree.write(dataset_new_path + "/" + ima_name + "_resiz" + ".xml")


#########水平镜像翻转处理及标签迁移
def xroll_image(ima_name , img_path , xml_path , save=False):
    
    ima_data = cv.imread(img_path)
    xroll_ima = cv.flip(ima_data,1,dst=None) 
    
    ima_box = []
    tree = ET.parse(xml_path)
    root = tree.getroot() 
    for box in root.iter('bndbox'):
        #xmin ymin xmax ymax 
        value = (float(box.find('xmin').text) , 
                 float(box.find('ymin').text) , 
                 float(box.find('xmax').text) , 
                 float(box.find('ymax').text))
        ima_box.append(value)
        
    xroll_box = []
    for i in range(len(ima_box)):
        box_value = ima_box[i]

        value = (xroll_ima.shape[1]-box_value[2] ,
                 box_value[1] ,
                 xroll_ima.shape[1]-box_value[0] ,
                 box_value[3])
     
        xroll_box.append(value)
    
    #更新TREE
    root.find("filename").text = ima_name + "_xroll" + ".jpg"
    root.find("path").text = dataset_new_path + "/" + ima_name + "_xroll" + ".jpg"

    #更新BOX
    idx = 0
    for box in root.iter('bndbox'):
        new_value = xroll_box[idx]
        idx += 1

        box.find('xmin').text = str(int(new_value[0]))
        box.find('ymin').text = str(int(new_value[1]))
        box.find('xmax').text = str(int(new_value[2]))
        box.find('ymax').text = str(int(new_value[3]))
        box.set("updated",'yes')

    if save:
        cv.imwrite(dataset_new_path + "/" + ima_name + "_xroll" + ".jpg" , xroll_ima)
        tree.write(dataset_new_path + "/" + ima_name + "_xroll" + ".xml")

######读取文件夹中图片及XML文件
for i in os.listdir(dataset_path):
    ima_name , ima_type = os.path.splitext(i)
    if ima_type == '.jpg':
        img_path = os.path.join(dataset_path, ima_name + '.jpg')
        xml_path = os.path.join(dataset_path, ima_name + '.xml')
        
        xroll_image(ima_name , img_path , xml_path , save=True)
        origi_image(ima_name , img_path , xml_path , save=True)
        resize_image(ima_name , img_path , xml_path , newsize=300 , save=True)
        



