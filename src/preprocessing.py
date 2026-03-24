"""
preprocessing.py
----------------
Handles all data preprocessing for the Thyroid Cancer Classification pipeline.
Includes image loading, augmentation, dataset splitting, and generator creation.
"""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 16
CLASSES    = ['benign', 'malignant']
SEED       = 42

random.seed(SEED)
np.random.seed(SEED)


def split_dataset(
    raw_dir: str,
    train_dir: str,
    val_dir: str,
    test_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
    seed: int          = SEED
) -> Dict[str, Dict[str, int]]:
    """
    Split raw dataset into train / val / test folders.

    Expects raw_dir to contain subfolders: benign/ and malignant/
    Creates mirrored structure under train_dir, val_dir, test_dir.

    Parameters
    ----------
    raw_dir     : path to raw dataset root
    train_dir   : destination for training images
    val_dir     : destination for validation images
    test_dir    : destination for test images
    train_ratio : proportion for training (default 0.70)
    val_ratio   : proportion for validation (default 0.15)
    seed        : random seed for reproducibility

    Returns
    -------
    dict with split counts per class
    """
    random.seed(seed)
    counts = {}

    for cls in CLASSES:
        src  = Path(raw_dir) / cls
        imgs = list(src.glob('*.jpg')) + list(src.glob('*.jpeg'))
        random.shuffle(imgs)

        n_train = int(len(imgs) * train_ratio)
        n_val   = int(len(imgs) * val_ratio)
        n_test  = len(imgs) - n_train - n_val

        splits = {
            train_dir: imgs[:n_train],
            val_dir:   imgs[n_train:n_train + n_val],
            test_dir:  imgs[n_train + n_val:]
        }

        for split_dir, split_imgs in splits.items():
            dest = Path(split_dir) / cls
            dest.mkdir(parents=True, exist_ok=True)
            for p in split_imgs:
                shutil.copy(p, dest / p.name)

        counts[cls] = {'train': n_train, 'val': n_val, 'test': n_test}
        print(f'{cls:12s} → train: {n_train}, val: {n_val}, test: {n_test}')

    return counts


def get_train_generator(train_dir: str) -> ImageDataGenerator:
    """
    Create augmented training data generator.

    Uses EfficientNet's native preprocess_input (NOT rescale=1./255).
    Augmentations are chosen for ultrasound image characteristics:
    - vertical_flip=True  : ultrasound images are orientation-agnostic
    - brightness_range    : simulates different ultrasound gain settings

    Parameters
    ----------
    train_dir : path to training directory with class subfolders

    Returns
    -------
    Keras DirectoryIterator (generator)
    """
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode='nearest'
    )
    return datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=SEED,
        shuffle=True
    )


def get_val_test_generator(data_dir: str, shuffle: bool = False):
    """
    Create validation or test data generator (no augmentation).

    Parameters
    ----------
    data_dir : path to val or test directory with class subfolders
    shuffle  : whether to shuffle (False for eval, True for viz)

    Returns
    -------
    Keras DirectoryIterator (generator)
    """
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    return datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=SEED,
        shuffle=shuffle
    )


def get_retrain_generators(new_data_dir: str, val_split: float = 0.15):
    """
    Create train + val generators from newly uploaded data for retraining.

    Parameters
    ----------
    new_data_dir : path to new data directory with class subfolders
    val_split    : proportion to use for validation

    Returns
    -------
    tuple of (train_generator, val_generator)
    """
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.25,
        validation_split=val_split
    )

    train_gen = datagen.flow_from_directory(
        new_data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        seed=SEED
    )
    val_gen = datagen.flow_from_directory(
        new_data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        seed=SEED
    )
    return train_gen, val_gen


def preprocess_single_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a single image for inference.

    Parameters
    ----------
    image_path : path to the image file (.jpg / .jpeg / .png)

    Returns
    -------
    numpy array of shape (1, 224, 224, 3) ready for model.predict()
    """
    img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load and preprocess an image from raw bytes (for API use).

    Parameters
    ----------
    image_bytes : raw image bytes from file upload

    Returns
    -------
    numpy array of shape (1, 224, 224, 3) ready for model.predict()
    """
    import io
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def compute_class_weights(generator) -> Dict[int, float]:
    """
    Compute balanced class weights from a generator.

    Parameters
    ----------
    generator : Keras DirectoryIterator

    Returns
    -------
    dict mapping class index to weight, e.g. {0: 1.2, 1: 0.8}
    """
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    cw = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=generator.classes
    )
    weights = {0: float(cw[0]), 1: float(cw[1])}
    print(f'Class weights: {weights}')
    return weights


def get_dataset_info(raw_dir: str) -> Dict:
    """
    Return basic statistics about the raw dataset.

    Parameters
    ----------
    raw_dir : path to raw dataset root

    Returns
    -------
    dict with counts, imbalance ratio, and class names
    """
    info = {}
    for cls in CLASSES:
        cls_dir = Path(raw_dir) / cls
        imgs    = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.jpeg'))
        info[cls] = len(imgs)

    total    = sum(info.values())
    majority = max(info.values())
    minority = min(info.values())
    ratio    = majority / (minority + 1e-8)

    return {
        'counts'          : info,
        'total'           : total,
        'imbalance_ratio' : round(ratio, 2),
        'is_imbalanced'   : ratio > 1.5,
        'class_names'     : CLASSES
    }


if __name__ == '__main__':
    # Quick test
    info = get_dataset_info('./data/raw')
    print('Dataset info:')
    for k, v in info.items():
        print(f'  {k}: {v}')