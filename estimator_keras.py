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
from IPython import embed
import h5py as h5

batch_size = 1000 
#num_classes = 9999 
num_classes = 14950
epochs = 10
LEARNING_RATE = 1e-4

# input image dimensions
img_rows, img_cols = 128, 128
TRAIN_PATH = "/scratch/vicente/65000_images.hdf5"
TEST_PATH = "/scratch/vicente/65000_images.hdf5"

def load_data():
    train = h5.File(TRAIN_PATH)

    dataset_y = train['Y'][:]
    dataset_x = train['X'][:]

    train_y = dataset_y[:50000]
    train_x = dataset_x[:50000]

    test_y = dataset_y[50000:60000]
    test_x = dataset_x[50000:60000]

    valid_y = dataset_y[60000:]
    valid_x = dataset_x[60000:]


    return (train_x, train_y), (test_x, test_y), (valid_x, valid_y)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) , (x_valid, y_valid)= load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    x_valid = x_valid.reshape(x_valid.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols,3)
    input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_valid = x_valid .astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
#model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(LEARNING_RATE),
              #optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          shuffle=True,
          epochs=epochs,
          verbose=1,
          validation_data=(x_valid, y_valid))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
embed()
