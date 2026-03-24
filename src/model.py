"""
model.py
--------
Handles EfficientNetB0 model creation, compilation, training, and retraining
for the Thyroid Cancer Classification pipeline.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau,
    ModelCheckpoint, CSVLogger
)

from preprocessing import (
    get_train_generator, get_val_test_generator,
    get_retrain_generators, compute_class_weights,
    IMG_SIZE, BATCH_SIZE, SEED
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR  = Path('./models')
MODEL_H5    = MODELS_DIR / 'thyroid_efficientnet.h5'
MODEL_TF    = MODELS_DIR / 'thyroid_efficientnet.tf'
BEST_CKPT   = MODELS_DIR / 'best_model.h5'
META_PATH   = MODELS_DIR / 'model_meta.pkl'
TRAIN_LOG   = MODELS_DIR / 'training_log.csv'
RETRAIN_LOG = MODELS_DIR / 'retrain_log.csv'

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS        = 40
LR            = 1e-4
UNFREEZE_LAST = 30      # number of EfficientNet layers to keep trainable
RETRAIN_THRESHOLD = 0.75  # auto-retrain if val accuracy drops below this


def build_model(
    img_size: Tuple[int, int] = IMG_SIZE,
    lr: float = LR,
    unfreeze_last: int = UNFREEZE_LAST
) -> keras.Model:
    """
    Build and compile EfficientNetB0 binary classifier (v3 — regularised).

    Architecture
    ------------
    EfficientNetB0 (top 30 layers trainable)
      → GlobalAveragePooling2D
      → BatchNormalization
      → Dropout(0.3)
      → Dense(128, relu, L2=0.02)
      → Dropout(0.6)
      → Dense(1, sigmoid)

    Parameters
    ----------
    img_size      : input image dimensions (H, W)
    lr            : initial learning rate
    unfreeze_last : number of EfficientNet layers to unfreeze from the top

    Returns
    -------
    Compiled Keras Model
    """
    inputs = keras.Input(shape=(*img_size, 3))

    base = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )

    # Partial unfreeze — top N layers trainable
    base.trainable = True
    for layer in base.layers[:-unfreeze_last]:
        layer.trainable = False

    trainable_count = sum(1 for l in base.layers if l.trainable)
    print(f'EfficientNetB0: {trainable_count}/{len(base.layers)} layers trainable')

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(
        128, activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.02)
    )(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs, name='ThyroidEfficientNetV3')

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=1e-4,
            clipnorm=1.0
        ),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )

    print(f'Model built: {model.name}')
    print(f'Total params: {model.count_params():,}')
    return model


def get_callbacks(checkpoint_path: str = str(BEST_CKPT),
                  log_path: str = str(TRAIN_LOG)) -> list:
    """
    Return standard training callbacks.

    Parameters
    ----------
    checkpoint_path : where to save best model weights
    log_path        : where to write CSV training log

    Returns
    -------
    list of Keras callbacks
    """
    return [
        EarlyStopping(
            monitor='val_auc', patience=8,
            restore_best_weights=True, mode='max', verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.3, patience=4,
            min_lr=1e-7, verbose=1
        ),
        ModelCheckpoint(
            checkpoint_path, monitor='val_auc',
            save_best_only=True, mode='max', verbose=1
        ),
        CSVLogger(log_path, append=True)
    ]


def train(
    train_dir: str,
    val_dir: str,
    epochs: int = EPOCHS
) -> Tuple[keras.Model, dict]:
    """
    Full training pipeline: build → train → save.

    Parameters
    ----------
    train_dir : path to training data directory
    val_dir   : path to validation data directory
    epochs    : maximum training epochs

    Returns
    -------
    (trained model, training history dict)
    """
    train_gen = get_train_generator(train_dir)
    val_gen   = get_val_test_generator(val_dir)
    cw        = compute_class_weights(train_gen)

    model = build_model()

    print(f'\nStarting training for up to {epochs} epochs...')
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=get_callbacks(),
        class_weight=cw,
        verbose=1
    )

    # Load best checkpoint and save final
    best = keras.models.load_model(str(BEST_CKPT))
    best.save(str(MODEL_H5))
    best.save(str(MODEL_TF))
    print(f'\n✅ Training complete. Model saved to {MODELS_DIR}')

    return best, history.history


def retrain(
    new_data_dir: str,
    model_path: str = str(MODEL_H5),
    epochs: int     = 10,
    lr: float       = 1e-5
) -> Tuple[keras.Model, dict]:
    """
    Retrain existing model on newly uploaded data.

    Called by:
      - POST /retrain API endpoint (user presses Retrain button in UI)
      - should_retrain() when accuracy drops below threshold

    Parameters
    ----------
    new_data_dir : path to new data with subfolders benign/ malignant/
    model_path   : path to saved model to resume from
    epochs       : max retraining epochs
    lr           : fine-tuning learning rate (lower than initial)

    Returns
    -------
    (retrained model, history dict)
    """
    print(f'Loading model from {model_path} ...')
    model = keras.models.load_model(model_path)

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=1e-4,
            clipnorm=1.0
        ),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )

    train_gen, val_gen = get_retrain_generators(new_data_dir)
    cw = compute_class_weights(train_gen)

    callbacks = [
        EarlyStopping(
            monitor='val_auc', patience=4,
            restore_best_weights=True, mode='max', verbose=1
        ),
        ModelCheckpoint(
            str(BEST_CKPT), monitor='val_auc',
            save_best_only=True, mode='max', verbose=1
        ),
        CSVLogger(str(RETRAIN_LOG), append=True)
    ]

    print(f'Retraining for up to {epochs} epochs...')
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=cw,
        verbose=1
    )

    best = keras.models.load_model(str(BEST_CKPT))
    best.save(str(MODEL_H5))
    best.save(str(MODEL_TF))
    print('✅ Retraining complete. Model updated.')

    return best, history.history


def load_model() -> keras.Model:
    """
    Load the saved production model from disk.

    Returns
    -------
    Loaded Keras Model
    """
    if not MODEL_H5.exists():
        raise FileNotFoundError(
            f'No saved model found at {MODEL_H5}. '
            'Run train() first.'
        )
    print(f'Loading model from {MODEL_H5} ...')
    return keras.models.load_model(str(MODEL_H5))


def save_model_meta(
    train_gen,
    metrics: Dict,
    optimal_threshold: float
) -> None:
    """
    Save model metadata (class names, metrics, threshold) as pickle.

    Parameters
    ----------
    train_gen          : training generator (for class_indices)
    metrics            : dict of evaluation metrics
    optimal_threshold  : best F1 decision threshold
    """
    meta = {
        'class_indices'    : train_gen.class_indices,
        'class_names'      : list(train_gen.class_indices.keys()),
        'img_size'         : IMG_SIZE,
        'optimal_threshold': float(optimal_threshold),
        'model_version'    : 'v3',
        'metrics'          : {k: float(v) for k, v in metrics.items()}
    }
    with open(str(META_PATH), 'wb') as f:
        pickle.dump(meta, f)
    print(f'✅ Model metadata saved to {META_PATH}')


def load_model_meta() -> Dict:
    """
    Load saved model metadata from disk.

    Returns
    -------
    dict with class_names, img_size, optimal_threshold, metrics
    """
    if not META_PATH.exists():
        raise FileNotFoundError(f'No metadata found at {META_PATH}.')
    with open(str(META_PATH), 'rb') as f:
        return pickle.load(f)


def should_retrain(recent_val_accuracy: float) -> bool:
    """
    Determine whether the model should be retrained based on recent performance.

    Trigger condition: val accuracy drops below RETRAIN_THRESHOLD.
    Can be called periodically from a monitoring job or after each batch
    of new predictions.

    Parameters
    ----------
    recent_val_accuracy : recent validation or production accuracy (0.0–1.0)

    Returns
    -------
    bool — True if retraining is recommended
    """
    trigger = recent_val_accuracy < RETRAIN_THRESHOLD
    if trigger:
        print(
            f'⚠️  Accuracy {recent_val_accuracy:.3f} < threshold '
            f'{RETRAIN_THRESHOLD} — retraining recommended.'
        )
    else:
        print(f'✅ Accuracy {recent_val_accuracy:.3f} above threshold. No retraining needed.')
    return trigger


if __name__ == '__main__':
    # Quick sanity check — build and summarise model
    m = build_model()
    m.summary()