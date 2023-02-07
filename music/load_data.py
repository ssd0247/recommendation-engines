"""Converts images and labels into training and testing matrices."""
import os
import re
import numpy as np
import cv2
from .import_data import create_spectogram
from .slice_spectogram import vs_method, kaggle_method
from sklearn.model_selection import train_test_split


def _to_categorical(x, num_classes):
    """One-Hot encode a tensor."""
    return np.eye(num_classes, dtype='unit8')[x]

def load_dataset(test_size=0.05, verbose=0, mode=None, datasetSize=1.0, vs_method=False, mel_spectogram=False):
    create_spectogram(verbose, mode)
    if vs_method:
        from .slice_spectogram import vs_method as vsm
        slice_spect = vsm()
        slice_spect(verbose, mode)
    else:
        from .slice_spectogram import kaggle_method
        kaggle_method(verbose, mel_spectogram)
    
    # datasetSize is a float value which returns a fraction of the dataset.
    # If set as 1.0 it returns the entire dataset.
    # If set as 0.5 it returns half the dataset.

    if mode == 'Train':
        genre = {
            "Hip-Hop": 0,
            "International": 1,
            "Electronic": 2,
            "Folk": 3,
            "Experimental": 4,
            "Rock": 5,
            "Pop": 6,
            "Instrumental": 7,
        }
        if verbose > 0:
            print("Compiling Training and Testing Sets ...")
        filenames = [os.path.join('data_collection/data/Train_Sliced_Images', f)
            for f in os.listdir('data_collection/data/Train_Slices_Images')
            if f.endswith(".jpg")]
        images_all = [None] * len(filenames)
        labels_all = [None] * len(filenames)
        for f in filenames:
            index = int(re.search('Train_Slices_Images/(.+?)_.*.jpg', f).group(1))
            genre_variable = re.search('Train_Sliced_Images/.*_(.+?).jpg', f).group(1)
            temp = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            images_all[index] = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            labels_all[index] = genre[genre_variable]
        
        if datasetSize == 1.0:
            images = images_all
            labels = labels_all
        else:
            count_max = int(len(images_all) * datasetSize / 8.0)
            count_array = [0] * 8
            images = []
            labels = []
            for i in range(len(images_all)):
                if count_array[labels_all[i]] < count_max:
                    images.append(images_all[i])
                    labels.append(labels_all[i])
                    count_array[labels_all[i]] += 1
            images = np.array(images)
            labels = np.array(labels)
        
        images = np.array(images)
        labels = np.array(labels)
        labels = labels.reshape(labels.shape[0], 1)
        train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=test_size, shuffle=True)
        
        # Convert the labels into one-hot vectors.
        n_classes = len(genre)
        genre_new = {value: key for key, value in genre.items()}
        train_y = _to_categorical(x=train_y, num_classes=n_classes)
        test_y = _to_categorical(x=test_y, num_classes=n_classes)

        path = 'data_collection/data/Training_Data'
        if os.path.exists(path):
            train_x = np.load(path + '/train_x.npy')
            train_y = np.load(path + '/train_y.npy')
            test_x = np.load(path + '/test_x.npy')
            test_y = np.load(path + '/test_y.npy')
            return train_x, train_y, test_x, test_y, n_classes, genre_new
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        np.save(path + '/train_x.npy', train_x)
        np.save(path + '/train_y.npy', train_y)
        np.save(path + '/test_x.npy', test_x)
        np.save(path + '/test_y.npy', test_y)
        return train_x, train_y, test_x, test_y, n_classes, genre_new
    
    if mode == 'Test':
        if verbose > 0:
            print("Compiling Training and Testing Sets...")
        filenames = [os.path.join('data_collection/data/Test_Slices_Images', f)
            for f in os.listdir('data_collection/data/Test_Sliced_Images')
            if f.endswith(".jpg")]
        
        images = []
        labels = []
        for f in filenames:
            song_variable = re.search('Test_Sliced_Images/.*_(.+?).jpg', f).group(1)
            tempImg = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            images.append(cv2.cvtColor(tempImg, cv2.COLOR_BGR2GRAY))
            labels.append(song_variable)

        images = np.array(images)

        return images, labels