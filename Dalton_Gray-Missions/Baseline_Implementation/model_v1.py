from keras.models import Sequential 
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam
from keras.utils import to_categorical

model = Sequential()

model.add(Conv2D(32,(3,3),input_shape = (100,100,3),activation = 'relu',padding='same',))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

datagen = ImageDataGenerator(rescale=1.0/255.0)

train = datagen.flow_from_directory('data/training',
                                    class_mode='binary',
                                    batch_size=64,
                                    target_size=(200,200))

test = datagen.flow_from_directory('data/validation',
                                    class_mode='binary',
                                    batch_size=64,
                                    target_size=(200,200))
history = model.fit_generator(train,
                              validation_data=(test),
                              epochs = 50,
                              steps_per_epoch=len(train),
                              validation_steps=len(test))


model.save("model_v1.h5")