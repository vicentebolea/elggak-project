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
import h5py

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
    Y = []
    LogFile = open("LogResize.txt","w")
    #X = np.empty((1,128,128,3))
    count = 1
    Temp = ""
    with open('train.csv') as csvfile:
        flag = True
        readCSV = csv.reader(csvfile, delimiter=',')
        #Read each csv path
        for row in readCSV:
           file = readpath+"/"+row[0]+".jpg"
           #file="TRAIN_DATA_RESIZE/28fe865fecb4ec15.jpg"
           print((str(count))+"/1190794")
           
           if os.path.isfile(file):
               count = count + 1;
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
               LogFile.write(row[0] + "\n")
           else:
               continue
           if(count % 2000) == 0:
               path="./"
               Temp1="savetemp_" + str(count)
               Y1= np.asarray(Y).astype(np.int64)
               print(("Saving :"+ str(count))+"/1190794")
               with h5py.File(os.path.join(path, Temp1), 'w') as hf:
                     hf.create_dataset('X', data=X, compression="gzip")
                     hf.create_dataset('Y', data=Y1, compression="gzip")
               if os.path.exists(Temp):
                   os.remove(Temp)  
               Temp="savetemp_" + str(count)            
               
        Y = np.asarray(Y).astype(np.int64)
    LogFile.close()
    return FileName, X, Y
readPath="TRAIN_DATA_RESIZE"
csvFile="train.csv"
path ="./"
FileName, X, Y = ReadData(csvFile,readPath)

#Saving data
#np.savez('Data',FileName=FileName,X=X,Y=Y)
with h5py.File(os.path.join(path, 'Data.h5'), 'w') as hf:
    hf.create_dataset('X', data=X, compression="gzip")
    hf.create_dataset('Y', data=Y, compression="gzip")