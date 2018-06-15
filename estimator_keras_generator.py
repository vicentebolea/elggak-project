'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
#import keras.utils.vis_utils as kutil
from IPython import embed
import h5py as h5

batch_size = 500 
#num_classes = 9999 
num_classes = 14950
epochs = 40 
LEARNING_RATE = 0.01

# input image dimensions
img_rows, img_cols = 128, 128
TRAIN_PATH = "/scratch2/vicente/H5_FILES/35318_31688.h5"

def coro_load_data(read_size,batch_size):
  train = h5.File(TRAIN_PATH)
  file_size = len(train['Y'][:])
  #t = train['Y'][:]
  #t = train['X'][:]
  limit = 0

  while limit < file_size:
    train_y = train['Y'][limit:limit+read_size]
    train_x = train['X'][limit:limit+read_size]

#    if K.image_data_format() == 'channels_first':
#      train_x = train_x.reshape(train_x.shape[0], 3, img_rows, img_cols)
#      input_shape = (3, img_rows, img_cols)
#    else:
    train_x = train_x.reshape(train_x.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

    train_x = train_x.astype('float32')

    # convert class vectors to binary class matrices
    train_y = keras.utils.np_utils.to_categorical(train_y, num_classes)

    #yield (train_x, train_y)
    i = 0 
    while i < len(train_y):

      yield (train_x[i:i+batch_size], train_y[i:i+batch_size])
      i += batch_size 

    limit += read_size 
    limit %= file_size


def load_data():
  train = h5.File(TRAIN_PATH)

  valid_y = train['Y'][0:280]
  valid_x = train['X'][0:280]

  test_y = train['Y'][5000:25000]
  test_x = train['X'][5000:25000]

  return (test_x, test_y), (valid_x, valid_y)

(x_test, y_test) , (x_valid, y_valid)= load_data()

#if K.image_data_format() == 'channels_first':
#  x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
#  x_valid = x_valid.reshape(x_valid.shape[0], 3, img_rows, img_cols)
#  input_shape = (3, img_rows, img_cols)
#else:
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols,3)
input_shape = (img_rows, img_cols, 3)

x_test = x_test.astype('float32')
x_valid = x_valid .astype('float32')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
y_valid = keras.utils.np_utils.to_categorical(y_valid, num_classes)

model = Sequential()
#model.add(Conv2D(32, kernel_size=(4, 4),
#                 activation='relu',
#                 input_shape=input_shape))
##model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.25))
#model.add(Dense(num_classes, activation='softmax'))

model.add(Conv2D(32, 4, 4, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, 4, 4, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(128, 4, 4, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(LEARNING_RATE),
    #optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy'])

#kutil.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


cb_tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=20, write_graph=True, write_images=False)

cb_ckpt = keras.callbacks.ModelCheckpoint('./logs/weights.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=1) #,save_best_only=True, save_weights_only=False, mode='auto', period=10)

model.fit_generator(coro_load_data(100000,100), 
    samples_per_epoch=100000,
    nb_epoch=epochs,
    #shuffle=True,
    verbose=1,
    validation_data=(x_valid, y_valid),
    callbacks=[cb_tensorboard, cb_ckpt])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
embed()
