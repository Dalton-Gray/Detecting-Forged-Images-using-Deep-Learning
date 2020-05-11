from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.optimizers import Adam
import json
from time import time 

img_width, img_height = 300, 300

train_data_dir = 'data/training'
validation_data_dir = 'data/validation'

#change
nb_train_samples = 4600
#change
nb_validation_samples = 1500
epochs = 10
#change
batch_size = 64

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              # change
              optimizer= Adam(lr=0.001),
              #optimizer='rmsprop',
              metrics=['accuracy']) 

print(model.summary())

# ImageDataGenerator allows us to transform data to produce more unique data for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    # change
    horizontal_flip=False,
    #change
    vertical_flip=False)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

start = time()
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
    #callbacks=[lr_callbacks])
end = time()
print('Processing time:',(end - start)/60)
# change:
scores = model.evaluate_generator(validation_generator,400) 
print("Accuracy = ", scores[1])
# This function saves:
# * model architecture, allowing easy re-creation of the model
# * the weights of the model
# * the training configuration (loss, optimizer)
# * the state of the optimizer, allows resuming of training exactly where left off
# lets assume `model` is main model 
model.save('model_FINAL.h5')
