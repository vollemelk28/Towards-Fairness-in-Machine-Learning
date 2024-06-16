import numpy as np
import os
import pandas as pd
from keras.utils import to_categorical
import tensorflow as tf
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

IMG_SIZE = 144

def load_and_process_data(directories, batch_size, split_ratio, limits=None):
    image_paths = []
    ethnicity_labels = []

    for directory_index, datadir in enumerate(directories):
        dir = os.listdir(datadir)
        random.shuffle(dir)

        if limits is None:
            limit = None
        else:
            limit = limits[directory_index]

        for i, filename in enumerate(dir):
            if limit is not None and i >= limit:
                break
            image_path = os.path.join(datadir, filename)
            temp = filename.split('_')
            ethnicity = int(temp[2])
            image_paths.append(image_path)
            ethnicity_labels.append(ethnicity)

    # convert to dataframe
    df = pd.DataFrame()
    df['image'], df['ethnicity'] = image_paths, ethnicity_labels
    
    # Split the data
    train_df, val_df = train_test_split(df, test_size=split_ratio)
    
    # convert ethnicity column to string
    train_df['ethnicity'] = train_df['ethnicity'].astype(str)
    val_df['ethnicity'] = val_df['ethnicity'].astype(str)
    
    # Initialize ImageDataGenerator
    datagen = ImageDataGenerator(rescale=1./255.)

    # fit the generator with your data
    train_generator = datagen.flow_from_dataframe(train_df, x_col='image', y_col='ethnicity', classes=['0', '1', '2', '3'], 
                                                  class_mode='categorical', target_size=(IMG_SIZE, IMG_SIZE), 
                                                  batch_size=batch_size)

    validation_generator = datagen.flow_from_dataframe(val_df, x_col='image', y_col='ethnicity', classes=['0', '1', '2', '3'], 
                                                       class_mode='categorical', target_size=(IMG_SIZE, IMG_SIZE), 
                                                       batch_size=batch_size)

    return train_generator, validation_generator

def load_fileloc(lowest_idx, add_training_type):

    white_utk = 'C:\\!Scriptie\\Thesiscode\\fotos\\utk_splitted\\white'
    black_utk = 'C:\\!Scriptie\\Thesiscode\\fotos\\utk_splitted\\black'
    asian_utk = 'C:\\!Scriptie\\Thesiscode\\fotos\\utk_splitted\\asian'
    indian_utk = 'C:\\!Scriptie\\Thesiscode\\fotos\\utk_splitted\\indian'

    white_data_aug = 'C:\\!Scriptie\\Thesiscode\\fotos\data_aug\\white'
    black_data_aug = 'C:\\!Scriptie\\Thesiscode\\fotos\data_aug\\black'
    asian_data_aug = 'C:\\!Scriptie\\Thesiscode\\fotos\data_aug\\asian'
    indian_data_aug = 'C:\\!Scriptie\\Thesiscode\\fotos\data_aug\\indian'

    white_gan = 'C:\\!Scriptie\\Thesiscode\\fotos\\ganfotos_splitted\\white'
    black_gan = 'C:\\!Scriptie\\Thesiscode\\fotos\\ganfotos_splitted\\black'
    asian_gan = 'C:\\!Scriptie\\Thesiscode\\fotos\\ganfotos_splitted\\asian'
    indian_gan = 'C:\\!Scriptie\\Thesiscode\\fotos\\ganfotos_splitted\\indian'

    datasets = {
        0: {"real": white_utk, "data_aug": white_data_aug, "gan": white_gan},
        1: {"real": black_utk, "data_aug": black_data_aug, "gan": black_gan},
        2: {"real": asian_utk, "data_aug": asian_data_aug, "gan": asian_gan},
        3: {"real": indian_utk, "data_aug": indian_data_aug, "gan": indian_gan}
    }

    if lowest_idx in datasets:
        newdata = datasets[lowest_idx].get(add_training_type)

    return newdata

def delete_labelled_data(df, label, n):
    label_df = df[df['ethnicity'] == label]
    print("label_df: ", label_df)
    if len(label_df) < n:
        print("Error: Not enough images to delete")
        exit(1)
    
    delete_rows = label_df.sample(n)
    df = df.drop(delete_rows.index)
    return df


def extract_features(images):
    features = []
    for image in images:
        img = tf.keras.utils.load_img(image, color_mode='rgb', target_size=(IMG_SIZE, IMG_SIZE))
        img = np.array(img, dtype=np.uint8)  # Convert to numpy array with the right data type
        features.append(img)
        
    features = np.array(features)
    return features
