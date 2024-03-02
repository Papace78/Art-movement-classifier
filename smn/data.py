import os

from colorama import Fore, Style
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf

from smn.preprocess import process_image


"""
Extract data from the dataset and returns image batches
"""



def get_data(image_dir = './data/wikiart/in_work') -> pd.DataFrame:

    #Returns pandas df with path + style of each images

    data_dict = {'path':[] ,'style': []}
    for style in os.listdir(image_dir):

        if style not in ['Rococo',
                         'Abstract_Expressionism',
                         'Cubism',
                         'Northern_Renaissance',
                         'Naive_Art_Primitivism']:

            for image in os.listdir(os.path.join(image_dir,style)):
                data_dict['style'].append(style)
                data_dict['path'].append(image)

    data_df = pd.DataFrame(data_dict)
    print(Fore.BLUE + f"\ndataset with {len(data_df['style'].unique())} labels" + Style.RESET_ALL)

    #add two new columns

    data_df['full_path'] = image_dir + '/' + data_df['style'] + '/' + data_df['path']
    style_to_label = {style : i for i, style in enumerate(data_df['style'].unique())}
    data_df['style_label'] = data_df['style'].map(style_to_label)


    return data_df



def extract_from_data(df: pd.DataFrame, images_per_style = 2000) -> pd.DataFrame:

    # returns dataframe with chosen amount of image per pre-selected style

    return df.groupby('style').sample(images_per_style)



def split_data(df: pd.DataFrame, train_size = 0.9, val_size = 0.1, random_state = 42) -> pd.DataFrame:

    #Returns train and test data

    X_trainval, X_test, y_trainval, y_test = train_test_split(df[['full_path']],
                                                                                    df['style_label'],
                                                                                    train_size = train_size,
                                                                                    random_state = 42)

    X_train, X_val, y_train, y_val = train_test_split(X_trainval,
                                                    y_trainval,
                                                    test_size = val_size,
                                                    random_state = 42)

    return X_train, X_val, X_test, y_train, y_val, y_test



def generate_generators(X_train, X_val, X_test, y_train, y_val, y_test, batch_size = 16):

    #Returns batches of chosen size

    data_train = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
    data_val = tf.data.Dataset.from_tensor_slices((X_val.values, y_val.values))
    data_test = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))

    #prefetch accelerates the process

    train = data_train.map(process_image).batch(batch_size=batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    val = data_val.map(process_image).batch(batch_size=batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    test = data_test.map(process_image).batch(batch_size=batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

    return train, val, test
