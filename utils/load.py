import os

import scipy.ndimage
import scipy.misc
import numpy as np
import pandas as pd
import skimage.transform
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from lasagne.utils import floatX

from settings import IMAGES_DIR, WALK_START_INDEX, IMAGE_SIZE


def get_img(img_name, img_size=IMAGE_SIZE):
    target_shape = (img_size, img_size, 3)
    img = scipy.ndimage.imread(img_name)  # x*x*3
    assert img.dtype == 'uint8', img_name
    if len(img.shape) == 2:
        img = scipy.misc.imresize(img, (img_size, img_size))
        img = np.asarray([img, img, img])
    else:
        if img.shape[2] > 3:
            img = img[:, :, :3]
        img = scipy.misc.imresize(img, target_shape)
        img = np.rollaxis(img, 2)
    if img.shape[0] != 3:
        print(img_name)
    return img/255.


def get_images():
    for class_label, directory_info in enumerate(os.walk(IMAGES_DIR), WALK_START_INDEX):
        dir_name, subdir_names, image_filenames = directory_info
        if "n" not in dir_name:
            continue
        for image_name in image_filenames[:20]:
            yield os.path.abspath(os.path.join(dir_name, image_name)), class_label


def encode_classes(class_labels):
    enc = OneHotEncoder()
    enc.fit(class_labels)
    return enc.transform(class_labels).toarray()


def load_dataset():
    data = np.array([])
    class_labels = list()

    for image_name, class_label in get_images():
        image = get_img(image_name)
        class_labels.append(class_label)
        data = np.append(data, image)

    y = encode_classes(pd.Series(class_labels).reshape(-1, 1))
    data = data.reshape(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test


def prep_image(fn, image_mean):
    im = scipy.ndimage.imread(fn, mode='RGB')

    # Resize so smallest dim = 256, preserving aspect ratio

    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]

    rawim = np.copy(im).astype('uint8')

    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]

    im = im - image_mean

    return rawim, floatX(im[np.newaxis])


def load_data(image_mean):
    # Load and preprocess the entire dataset into numpy arrays
    X = []
    y = []

    for image_name, class_label in get_images():
        _, im = prep_image(image_name, image_mean)
        X.append(im)
        y.append(class_label)

    X = np.concatenate(X)
    y = np.array(y).astype('int32')
    return X, y


def load_dataset_transfer(image_mean):

    X, y = load_data(image_mean)
    # Split into train, validation and test sets
    train_ix, test_ix = train_test_split(range(len(y)))
    train_ix, val_ix = train_test_split(train_ix)

    X_train = X[train_ix]
    y_train = y[train_ix]

    X_val = X[val_ix]
    y_val = y[val_ix]

    X_test = X[test_ix]
    y_test = y[test_ix]

    return X_train, y_train, X_val, y_val, X_test, y_test
