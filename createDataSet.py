# -*- coding: utf-8 -*-
"""
Created on Sat May 26 00:24:23 2018

@author: kaka
"""

import numpy as np
import os.path
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import  array_to_img, img_to_array
import csv
def MyResize(img):
    mysize = 128
    img.thumbnail((mysize,mysize), Image.ANTIALIAS)
    width, height = img.size
    new_im = Image.new('RGB',(128,128))
    temp = width - height
    if temp > 0:
        new_im.paste(img,(0,(temp+1)//2))
    elif temp < 0:
        temp = temp*(-1)
        new_im.paste(img,((temp+1)//2,0))
    else:
        new_im.paste(img,(0,0))
    return new_im
def ReadData(csvFile,readpath):
    FileName=[]
    Y = []
    #X = np.empty((1,128,128,3))
    count = 0
    with open('train.csv') as csvfile:
        flag = True
        readCSV = csv.reader(csvfile, delimiter=',')
        #Read each csv path
        for row in readCSV:
           file = readpath+"/"+row[0]+".jpg"
           #file="TRAIN_DATA_RESIZE/28fe865fecb4ec15.jpg"
           #if count == 10:
           #    break
           print(("iterator:"+ str(count))+"/1190794")
           if os.path.isfile(file):
               img = Image.open(file).convert('RGB')
               img = MyResize(img)
               img = img_to_array(img)
               TempImg = img[np.newaxis,:,:,:]
               if flag:
                   X = TempImg
                   flag = False
               else:
                   X = np.append(X , TempImg , axis = 0)
               Y.append(row[2])   
               FileName.append(row[0])
               count = count + 1
           else:
               continue  
            #print(row[0],row[1],row[2],)
            #return
        Y = np.asarray(Y).astype(np.int64)
        FileName = np.asarray(FileName).astype(np.str)
    return FileName, X, Y
readPath="TRAIN_DATA_RESIZE"
csvFile="train.csv"
path ="./"
FileName, X, Y = ReadData(csvFile,readPath)

#Saving data
np.savez('savetest',FileName=FileName,X=X,Y=Y)
