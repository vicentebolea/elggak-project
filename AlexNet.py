from keras import  backend as K
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input,merge,Activation
from keras.optimizers import SGD,Adam
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.regularizers import l2
import numpy as np
import time
import h5py
import os.path
import pdb
from PIL import Image
import json
from keras.preprocessing.image import  ImageDataGenerator,array_to_img, img_to_array

def AlexNet(weights_path=None):

    inputs = Input(shape=(3,227,227))

    conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    conv_2 = merge([
        Convolution2D(128,5,5,activation="relu",name='conv_2_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = merge([
        Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = merge([
        Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_5)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu',name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096,   activation='relu',name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000,name='dense_3')(dense_3)
    prediction = Activation("softmax",name="softmax")(dense_3)

    model = Model(input=inputs, output=prediction)

    if weights_path:
        model.load_weights(weights_path)
        print 'Loaded weights succeed'


    print 'Loaded model succeed'
    return model

def MyTrainGenerator(X,Y,batch_size = 64, modeFineTune = 1):

    train_datagen = ImageDataGenerator(rotation_range = 90,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        shear_range = 0.2,
                                        zoom_range = 0.2,
                                        horizontal_flip = True,
                                        vertical_flip = True,
                                        fill_mode = 'nearest')

    if modeFineTune == 0:
        train_datagen = ImageDataGenerator(samplewise_center = True,
                                            samplewise_std_normalization = True,
                                            rescale = 1. / 255,
                                            rotation_range = 90,
                                            width_shift_range = 0.2,
                                            height_shift_range = 0.2,
                                            shear_range = 0.2,
                                            zoom_range = 0.2,
                                            horizontal_flip = True,
                                            vertical_flip = True,
                                            fill_mode = 'nearest')

    while (1):
        for X_batch, Y_batch in train_datagen.flow(X, Y, batch_size=batch_size):
            yield (X_batch,Y_batch)


def MyTestGenerator(X,Y,batch_size = 64, modeFineTune = 1):

    test_datagen = ImageDataGenerator()

    if modeFineTune = 0:

        test_datagen = ImageDataGenerator(samplewise_center = True,
                                            samplewise_std_normalization = True,
                                            rescale = 1. / 255)

    while (1):
        for X_batch, Y_batch in test_datagen.flow(X, Y, batch_size=batch_size):
            yield (X_batch,Y_batch)


if __name__ == "__main__":

    batch_size = 64
    nb_epoch = 200
    nb_classes = 5

    path = 'F:\Phat\VietNameseFoodCropped'
    path_weights = 'F:\Phat\\alexnet_weights.h5'
    modeFineTune = 1

    h5f = h5py.File(os.path.join(path, 'DataAlex.h5'), 'r')

    X_train = h5f['X_train'][:]
    Y_train = h5f['Y_train'][:]

    X_val = h5f['X_val'][:]
    Y_val = h5f['Y_val'][:]

    X_test = h5f['X_test'][:]
    Y_test = h5f['Y_test'][:]

    if modeFineTune == 1:

        X_train[:, 0, :, :] -= 123.68
        X_train[:,1,:,:] -= 116.779
        X_train[:,2,:,:] -= 103.939
        
        X_val[:, 0, :, :] -= 123.68
        X_val[:,1,:,:] -= 116.779
        X_val[:,2,:,:] -= 103.939
        
        X_test[:, 0, :, :] -= 123.68
        X_test[:,1,:,:] -= 116.779
        X_test[:,2,:,:] -= 103.939

    start_time = time.time()

    alexNet = AlexNet()
    myInput = alexNet.input

    finetune = alexNet.get_layer("dense_2").output
    finetune = Dropout(0.5)(finetune)
    classifiers = Dense(nb_classes, W_regularizer=l2(0.01), b_regularizer=l2(0.01), activation='softmax',
                                name='output')(finetune)

    model = Model(input=myInput, output=classifiers)

    checkpointer = ModelCheckpoint(filepath=os.path.join(temp, 'weights.{epoch:03d}.h5'),
                                       verbose=0, save_best_only=True)
    csv_logger = CSVLogger(filename=os.path.join(temp, 'Train.log'))

    lr = 1e-3
    decay = lr

    sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.fit_generator(MyTrainGenerator(X_train, Y_train, batch_size, modeFineTune),
                        samples_per_epoch=len(X_train),
                        verbose=0, nb_epoch=nb_epoch,
                        validation_data=MyTestGenerator(X_val, Y_val),
                        nb_val_samples=len(X_val),
                        callbacks=[checkpointer, csv_logger])

    score = model.evaluate_generator(MyTestGenerator(X_test, Y_test,batch_size,modeFineTune),
                                    val_samples=len(X_test))
        
    print score
    print "--- %s seconds ---" % (time.time() - start_time)
