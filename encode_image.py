import numpy as np
import os.path
from PIL import Image
#import matplotlib.pyplot as plt
from keras.preprocessing.image import  array_to_img, img_to_array
import csv
import sys
import h5py

def ReadData(files, level):
  Y = []
  X = None
  Temp = ""
  flag = True
  for img in files:
    if os.path.isfile(img):
      out = Image.open(img).convert('RGB')
      out = img_to_array(out)
      TempImg = out[np.newaxis,:,:,:]

      if flag:
        X = TempImg
        flag = False
      else:
        X = np.append(X , TempImg , axis = 0)

      Y.append(img.rsplit('/')[level])   

    else:
      continue

  Y = np.asarray(Y).astype(np.int64)
  return X, Y


X, Y = ReadData(sys.argv[3:], int(sys.argv[1]))

with h5py.File(sys.argv[2], 'w') as hf:
    hf.create_dataset('X', data=X, compression="gzip")
    hf.create_dataset('Y', data=Y, compression="gzip")
