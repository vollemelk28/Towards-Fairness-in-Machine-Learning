import time
import numpy as np
import os
from typing import List, Tuple
from matplotlib.pyplot import imshow
import json
import csv
import time
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

import matplotlib.pyplot as plt
from tensorflow import keras

from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.initializers import glorot_uniform
from keras_facenet import FaceNet
from tensorflow.keras.models import Model, load_model

from tensorflow.python.keras.utils import layer_utils
from tensorflow.keras.utils import model_to_dot

from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.imagenet_utils import preprocess_input

# Ethnicity split in UTK train set: 
# 8063
# 3617
# 2737
# 3139

# Variables for experimenting - Start

model_epochs = 10
retrain_epochs = 0
add_training_times = 0
add_training_type = "gan"           # "real" for real data, "data_aug" for augmented, "gan" for GAN
averaging = 10
limit = 100000                       # Amount of photos used for training. Any number above the max amount of images will default to that number.
extratrain = False                   # For additional training after initial model is trained.
add_pictures = 1000
manual_add = False                   # To add pictures manually without the decision of the model
lowest_idx = 3                       # Which class to train extra
adding_data = False                  # To add data for the initial training
testset = 1                          # 1 for UTK, 2 for FF
plot = True

# Variables for experimenting - End

import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')
import cv2
from typing import List, Tuple
import random
import gc

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
from data_loader import load_and_process_data
from data_loader import load_fileloc

from test_model import test_mod
from keras.models import load_model

# At PC 1
#BASE_DIR = 'C:\\Users\\a.schaap.student\\Documents\\UTK\\Thesiscode\\fotos\\utkzonderother'
#BASE_DIR = 'C:\\Users\\a.schaap.student\\Documents\\UTK\\Thesiscode\\fotos\\ff_train16k'
#BASE_DIR = 'C:\\Users\\a.schaap.student\\Documents\\UTK\\Thesiscode\\fotos\\utk_ff_eqreal16k'
#BASE_DIR = 'C:\\Users\\a.schaap.student\\Documents\\UTK\\Thesiscode\\fotos\\undersample_utk'
# At PC 2
# BASE_DIR = 'C:\\!Scriptie\\Thesiscode\\fotos\\utk+daug'                   # Data Augmentation
BASE_DIR = 'C:\\!Scriptie\\Thesiscode\\fotos\\utkzonderother'               # Baseline
# BASE_DIR = 'C:\\!Scriptie\\Thesiscode\\fotos\\undersample_utk'            # Undersample
# BASE_DIR = 'C:\\!Scriptie\\Thesiscode\\fotos\\utk_ff_eqreal16k'           # Oversample

IMG_SIZE = 144
batch_size = 8

acc_results = []
val_acc_results = []
loss_results = []
val_loss_results = []

if adding_data == True:

    newdata = load_fileloc(lowest_idx, add_training_type)

    X_new, y_ethnicity_onehot_new, df_new = load_and_process_data(newdata, add_pictures)

    X = np.concatenate((X, X_new), axis=0)

    y_ethnicity_onehot = np.concatenate((y_ethnicity_onehot_new, y_ethnicity_onehot_new), axis=0)
    df = np.concatenate((df, df_new), axis=0)

avg_eqodds = []
for step in range(averaging):
    print("This is step ", step, "...")
    tf.keras.backend.clear_session()
    
    ########################################################################
    ethnicity_dict = {0:'White', 1:'Black', 2:'Asian', 3:'Indian', 4:'Others'}

    if plot:
        plt.figure(figsize=(20, 20))
        files = df.iloc[200:225]

        for index, file, ethnicity in files.itertuples():

            plt.subplot(5, 5, index+1-200)
            img = tf.keras.utils.load_img(file)
            img = np.array(img)
            plt.imshow(img)
            plt.title(f"Ethnicity: {ethnicity_dict[ethnicity]}")
            plt.axis('off')

        plt.show()

    #########################################################################

    learning_rate = tf.Variable(0.001, trainable=True)
    tf.keras.backend.set_value(learning_rate, 0.000001)


    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    facenet_model = FaceNet()

    # Freeze first 10 layers of pretrained model
    for layer in facenet_model.model.layers[:-10]:
        layer.trainable = False


    input_tensor = Input(shape=input_shape)
    facenet_output = facenet_model.model(input_tensor)
    flatten_layer = Flatten()(facenet_output)
    dense_layer = Dense(256, activation='relu')(flatten_layer)
    dense_layer2 = Dense(128, activation='relu')(dense_layer)
    batch_norm_layer = BatchNormalization()(dense_layer2)
    output_layer = Dense(4, activation='softmax', name='ethnicity_output')(batch_norm_layer)

    model = tf.keras.Model(inputs=input_tensor, outputs=output_layer)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    directories = [BASE_DIR]
    train_generator, validation_generator = load_and_process_data(directories, batch_size, 0.2)
    steps_per_epoch = len(train_generator)
    validation_steps = len(validation_generator)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    score = model.evaluate_generator(train_generator, steps=steps_per_epoch, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps, 
    epochs=model_epochs
    )

    if plot:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        acc_results.append(acc)
        val_acc_results.append(val_acc)
        loss_results.append(loss)
        val_loss_results.append(val_loss)

    converged = False
    times_runned = 0
    i = 0

    list_lowest = []
    lowest_idx, converged, all_eqodds, all_acc, all_f1, all_di = test_mod(model, testset)

    while extratrain and not converged:

        i = i + 1
        print("Iteration: ", i)
        
        input_tensor = Input(shape=input_shape)
        facenet_output = facenet_model.model(input_tensor)
        flatten_layer = Flatten()(facenet_output)
        dense_layer = Dense(256, activation='relu')(flatten_layer)
        dense_layer2 = Dense(128, activation='relu')(dense_layer)
        batch_norm_layer = BatchNormalization()(dense_layer2)
        output_layer = Dense(4, activation='softmax', name='ethnicity_output')(batch_norm_layer)

        model = tf.keras.Model(inputs=input_tensor, outputs=output_layer)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        newdata = load_fileloc(lowest_idx, add_training_type)
        limit = add_pictures

        print("Lowest ID: ", lowest_idx)
        list_lowest.append(lowest_idx)
        if add_pictures != 0:
            dirs = [BASE_DIR, newdata]
            limits = [None, i * add_pictures]
            train_generator, validation_generator = load_and_process_data(dirs, batch_size, 0.2)
            steps_per_epoch = len(train_generator)
            validation_steps = len(validation_generator)

        history = model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps, 
            epochs=model_epochs
        )

        model.summary()

        if plot:
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            acc_results.append(acc)
            val_acc_results.append(val_acc)
            loss_results.append(loss)
            val_loss_results.append(val_loss)

        lowest_idx, converged, all_eqodds, all_acc, all_f1, all_di = test_mod(model, testset)

        if i == add_training_times:
            print(i, " training passed...")
            converged = True

    print("List lowest: ", list_lowest)
    

print("All accuracy: ", all_acc)
print("All DI: ", all_di)
print("All F1: ", all_f1)

np_data = np.array(all_eqodds)
reshaped_data = np_data.reshape(-1, i + 1, 4)
averaged_data = reshaped_data.mean(axis=0)
averaged_list = averaged_data.tolist()
print("all Averaged eqodds:", averaged_list)

# Calculate averages
avg_acc = sum(all_acc) / len(all_acc)
avg_di = sum(all_di) / len(all_di)  
avg_f1 = np.mean(np.concatenate(all_f1))
avg_eqodds = np.mean(np.concatenate(all_eqodds))
print("avg_acc: ", avg_acc)
print("avg_di: ", avg_di)
print("avg_f1: ", avg_f1)
print("avg_eqodds: ", avg_eqodds)

# Write to CSV file
filename = "LOCATION" + time.strftime("%Y%m%d-%H%M%S") + ".csv"
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["model_epochs", "retrain_epochs", "add_training_times", "add_training_type", "averaging", "limit", "extratrain", "add_pictures", "manual_add", "lowest_idx", "adding_data"])
    writer.writerow([model_epochs, retrain_epochs, add_training_times, add_training_type, averaging, limit, extratrain, add_pictures, manual_add, lowest_idx, adding_data])
    writer.writerow([])
    writer.writerow(["Date", "All accuracy", "All DI", "All F1", "Averaged eqodds", "avg_acc", "avg_di", "avg_f1", "avg_eqodds"])
    writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), all_acc, all_di, all_f1, averaged_list, avg_acc, avg_di, avg_f1, avg_eqodds])

# Reshape to map the results over all runs
acc_results = np.array(acc_results).reshape(-1, 10)
val_acc_results = np.array(val_acc_results).reshape(-1, 10)
loss_results = np.array(loss_results).reshape(-1, 10)
val_loss_results = np.array(val_loss_results).reshape(-1, 10)

acc_avg = np.mean(acc_results, axis=0)
val_acc_avg = np.mean(val_acc_results, axis=0)
loss_avg = np.mean(loss_results, axis=0)
val_loss_avg = np.mean(val_loss_results, axis=0)

acc_std = np.std(acc_results, axis=0)
val_acc_std = np.std(val_acc_results, axis=0)
loss_std = np.std(loss_results, axis=0)
val_loss_std = np.std(val_loss_results, axis=0)

if plot:
    epochs = range(1, 11)
    plt.xlim(1,10)

    # Accuracy plot including std deviation
    plt.fill_between(epochs, acc_avg - acc_std, acc_avg + acc_std, color='b', alpha=0.1)
    plt.fill_between(epochs, val_acc_avg - val_acc_std, val_acc_avg + val_acc_std, color='r', alpha=0.1)
    plt.plot(epochs, acc_avg, 'b', label='Average training accuracy')
    plt.plot(epochs, val_acc_avg, 'r', label='Average validation accuracy')
    plt.title('Average accuracy over 10 runs')
    plt.legend()
    plt.figure()
    plt.show()

    epochs = range(1, 11)
    plt.xlim(1,10)
    # Loss plot inclusing std deviation
    plt.fill_between(epochs, loss_avg - loss_std, loss_avg + loss_std, color='b', alpha=0.1)
    plt.fill_between(epochs, val_loss_avg - val_loss_std, val_loss_avg + val_loss_std, color='r', alpha=0.1)
    plt.plot(epochs, loss_avg, 'b', label='Average training loss')
    plt.plot(epochs, val_loss_avg, 'r', label='Average validation loss')
    plt.title('Average loss over 10 runs')
    plt.legend()
    plt.show()


    all_avg_eq_odds = np.array(averaged_list).reshape(10, 10)

    mean_eq_odds = all_avg_eq_odds.mean(axis=0)
    std_eq_odds = all_avg_eq_odds.std(axis=0)

    epochs = range(1, 11)

    # Equalized odds plot
    plt.fill_between(epochs, mean_eq_odds - std_eq_odds, mean_eq_odds + std_eq_odds, color='b', alpha=0.1)
    plt.plot(epochs, mean_eq_odds, 'b', label='Average equalized odds')
    plt.xticks(range(1, 11))
    plt.title('Average equalized odds when using additional training over 10 epochs')
    plt.legend()
    plt.show()