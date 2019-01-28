
import math, json, os, sys

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image

number_of_epochs = 1
learning_rate = 0.001
image_size = 256
DATA_DIR = 'Data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
#VALID_DIR = os.path.join(DATA_DIR, 'valid')
SIZE = (image_size, image_size)
BATCH_SIZE = 16


if __name__ == "__main__":
    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    #num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
    #num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

    gen = keras.preprocessing.image.ImageDataGenerator()
    #val_gen = keras.preprocessing.image.ImageDataGenerator()

    batches = gen.flow_from_directory(TRAIN_DIR, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    #val_batches = val_gen.flow_from_directory(VALID_DIR, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)

    model = keras.applications.resnet50.ResNet50()

    classes = list(iter(batches.class_indices))
    model.layers.pop()
    for layer in model.layers:
        layer.trainable=False
    last = model.layers[-1].output
    x = Dense(len(classes), activation="softmax")(last)
    finetuned_model = Model(model.input, x)
    finetuned_model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    early_stopping = EarlyStopping(patience=10)
    checkpointer = ModelCheckpoint('resnet50_best.h5', verbose=1, save_best_only=True)

    finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=number_of_epochs, callbacks=[early_stopping, checkpointer])#, validation_data=val_batches, validation_steps=num_valid_steps)
    finetuned_model.save('resnet50_final.h5')