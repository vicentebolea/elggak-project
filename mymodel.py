import numpy as np
import tensorflow as tf
import h5py as h5
import keras
import keras.backend as K
from keras.layers import *
from keras.models import Model

NUM_TRAIN=50000
NUM_VALID=5000

LEARNING_RATE = 1e-4
BATCH_SIZE = 1000
EPOCHS = 200


def get_data(path):
	data = h5.File(path)
	NUM_TEST = data['X'].shape[0] - NUM_TRAIN - NUM_VALID
	Y = data['Y'][:]
	Y = keras.utils.to_categorical(np.array(Y).astype(int))
	x_train = data['X'][0:NUM_TRAIN]
	y_train = Y[0:NUM_TRAIN,:]

	x_valid = data['X'][NUM_TRAIN:NUM_TRAIN+NUM_VALID]
	y_valid = Y[NUM_TRAIN:NUM_TRAIN+NUM_VALID,:]


	x_test = data['X'][NUM_TRAIN+NUM_VALID:]
	y_test = Y[NUM_TRAIN+NUM_VALID:,:]
	return (x_train, y_train, 
			x_valid, y_valid,
			x_test, y_test)

def my_model(feature_shape, num_class):
	#feature_data 
	inputs = Input(feature_shape, name = "feature")
	
	with tf.name_scope('Conv2D-Dense-1'):
		feature = Conv2D(96, 16, 16, subsample=(4,4), activation='relu')(inputs)
		feature = MaxPooling2D((3,3), strides=(2,2))(feature)
		feature = Flatten()(feature)
	with tf.name_scope('Dense'):
		cat = Dense(150, activation='relu')(feature)
		cat = Dropout(0.35)(cat)

		cat = Dense(50, activation='relu')(cat)
		cat = Dropout(0.2)(cat)
	
	outputs = Dense(num_class, activation='softmax', name='logits')(cat)
	model = Model(inputs=inputs, outputs=outputs)
	return model
[x_train, y_train, x_valid, y_valid, x_test, y_test] = get_data("./savetemp_65000")

print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)
print(x_test.shape)
print(y_test.shape)


feature_shape = (x_train.shape[1],x_train.shape[2],x_train.shape[3])
num_class = y_train.shape[1]

#Create model
model = my_model(feature_shape, num_class)

# Compile model with Adam optimizer
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(LEARNING_RATE),
              metrics=['accuracy'])


#Train model
model.fit(x_train,y_train, shuffle=True, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_valid,y_valid))
