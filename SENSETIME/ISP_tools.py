import numpy as np
import sys
import imageio

def pic2bayer(infile,outfile=""):
    '''
    translate image(png/bmp etc) to bayer patten raw image
    the bayer is fix to :
    B G
    G R
    to make it can save in normal image,
    extend single channal to 3, this will get a grayscale image like
    BBB-GGG
    GGG-RRR
    '''
    im = imageio.imread(infile)
#print(im)
    h,w,ch = im.shape
#print(im.dtype)
#    print(im.shape)
    for i in range(0,h):
        for j in range(0,w):
            if((i%2 == 0) & (j%2 == 0)):
                im[i,j,:] = im[i,j,2]   #B
            elif((i%2 == 0) & (j%2 == 1)):
                im[i,j,:] = im[i,j,1];  #G
            elif((i%2 == 1) & (j%2 == 0)):
                im[i,j,:] = im[i,j,1];  #G
            elif((i%2 == 1) & (j%2 == 1)):
                im[i,j,:] = im[i,j,0];  #R
    if(outfile != ""):
        imageio.imwrite(outfile,im)
    return im

def demosaic_nn(raw_img):
    '''
        nearest neighbor demosaic:
        B0 G1
        G2 R3
        fill the missing color channel use the nearest 2x2 color
        ->
        B0-G1-R3,B0-G1-R3
        B0-G2-R3,B0-G2-R3
    '''
    raw_img = raw_img.astype(np.uint16)
    h,w,_ = raw_img.shape
    rgb_img = np.zeros((h,w,3),dtype=np.uint16)
    for i in range(0,int(h/2)):
        for j in range(0,int(w/2)):
            b0 = raw_img[2*i,2*j,0];
            g1 = raw_img[2*i,2*j+1,0];
            g2 = raw_img[2*i+1,2*j,0];
            r3 = raw_img[2*i+1,2*j+1,0];
            rgb_img[2*i,2*j,:] = [r3,g1,b0];
            rgb_img[2*i,2*j+1,:] = [r3,g1,b0];
            rgb_img[2*i+1,2*j,:] = [r3,g2,b0];
            rgb_img[2*i+1,2*j+1,:] = [r3,g2,b0];
    return rgb_img.astype(np.uint8);

def demosaic_hs(raw_img):
    '''
        half size demosaic :
        B00 G01 B02 G03
        G10 R11 G12 R13
        B20 G21 B22 G23
        G30 R31 G32 R33
        2X2->1 rgb pixel
        (B00,(G01+G10)/2,R11),(B02,(G03+G12)/2,R13)
        (B20,(G21+G30)/2,R31),(B22,(G23+G32)/2,R33)
    '''
    raw_img = raw_img.astype(np.uint16)   # to avoid uint8 overflow
    h,w,_=raw_img.shape

    h = int(h/2)
    w = int(w/2)
    rgb_img = np.zeros((h,w,3),dtype=np.uint16);
    for i in range(0,h):
        for j in range(0,w):
            b = raw_img[2*i,2*j,0];
            g = (raw_img[2*i+1,2*j,0] + raw_img[2*i,2*j+1,0])/2
            r = raw_img[2*i+1,2*j+1,0];
            rgb_img[i,j,:] = [r,g,b];
    return rgb_img.astype(np.uint8)