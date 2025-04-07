import os
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential, Input
from keras.src.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback, ReduceLROnPlateau
from keras.src.layers import MaxPooling2D
from keras.src.layers import SeparableConv2D, GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.src.losses import Huber
from keras.src.optimizers import Adam
from keras.src.optimizers.schedules import CosineDecayRestarts
from matplotlib import image as mpimg
from sklearn.model_selection import train_test_split

df = pd.read_csv("cleaned_metadata.csv")


def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (160, 120))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    return img


image_paths = df["full_path"].values
labels = df[["x", "y", "theta"]].values

X_train_paths, X_test_paths, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
X_test = np.array([load_image(path) for path in X_test_paths])


def augment_image(image):
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.image.random_jpeg_quality(image, 60, 100)
    return image


def process_path(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, [120, 160])
    image = tf.cast(image, tf.float32) / 255.0
    image = augment_image(image)
    return image, label


batch_size = 128
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_paths, y_train))
train_dataset = train_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test_paths, y_test))
test_dataset = test_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

checkpoint_folder = "ckpt"
os.makedirs(checkpoint_folder, exist_ok=True)

batches_per_epoch = len(X_train_paths) // batch_size
first_decay_steps = 5 * batches_per_epoch

sgdr_scheduler = CosineDecayRestarts(
    initial_learning_rate=0.01,
    first_decay_steps=first_decay_steps,
    t_mul=2.0,
    m_mul=0.5,
    alpha=1e-5
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

optimizer = Adam(learning_rate=sgdr_scheduler)

model = Sequential([
    Input((120, 160, 1)),

    SeparableConv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    GlobalAveragePooling2D(),

    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(3, activation='linear')
])

model.compile(optimizer=optimizer, loss=Huber(delta=1.0))

log_path_file = "logs/latest_run_path.txt"
if not os.path.exists("logs"):
    os.makedirs("logs")

if os.path.exists(log_path_file):
    with open(log_path_file, "r") as f:
        log_dir = f.read().strip()
else:
    log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(log_path_file, "w") as f:
        f.write(log_dir)

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

background_img = mpimg.imread("assets/gametable.png")
image_height, image_width, _ = background_img.shape


def log_predictions(epoch, logs):
    points = 10
    arrow_length = 10

    y_pred = model.predict(X_test)
    # y_pred = (y_pred * df[["x", "y", "theta"]].std().values) + df[["x", "y", "theta"]].mean().values
    # y_test_scaled = (y_test * df[["x", "y", "theta"]].std().values) + df[["x", "y", "theta"]].mean().values
    y_test_scaled = y_test

    plt.figure(figsize=(8, 6))

    plt.imshow(background_img, extent=[0, image_width, image_height, 0], aspect='auto', alpha=0.9)

    plt.scatter(y_pred[:points, 0], y_pred[:points, 1], alpha=0.7, s=100, c='red', label='Predicted')
    plt.scatter(y_test_scaled[:points, 0], y_test_scaled[:points, 1], alpha=0.7, s=100, c='blue', label='Actual')

    for i in range(points):
        plt.plot([y_test_scaled[i, 0], y_pred[i, 0]], [y_test_scaled[i, 1], y_pred[i, 1]], 'k--', alpha=0.6)

        dx_pred = arrow_length * np.cos(y_pred[i, 2])
        dy_pred = arrow_length * np.sin(y_pred[i, 2])
        plt.arrow(y_pred[i, 0], y_pred[i, 1], dx_pred, dy_pred, head_width=5, head_length=5, fc='green', ec='green')

        dx_test = arrow_length * np.cos(y_test_scaled[i, 2])
        dy_test = arrow_length * np.sin(y_test_scaled[i, 2])
        plt.arrow(y_test_scaled[i, 0], y_test_scaled[i, 1], dx_test, dy_test, head_width=5, head_length=5, fc='green',
                  ec='green')

    plt.xlim(-image_width * 0.1, image_width * 1.1)
    plt.ylim(image_height * 1.1, -image_height * 0.1)

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Predictions vs Actuals - Epoch {epoch}")

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{log_dir}/epoch_{epoch}.png")
    plt.close()

    file_writer = tf.summary.create_file_writer(log_dir)
    with file_writer.as_default():
        img = tf.io.read_file(f"{log_dir}/epoch_{epoch}.png")
        img = tf.image.decode_png(img)
        tf.summary.image("Predictions", tf.expand_dims(img, 0), step=epoch)

    with open(os.path.join(checkpoint_folder, "epoch.txt"), "wb") as f:
        f.write(str(epoch).encode())


predictions_callback = LambdaCallback(on_epoch_end=log_predictions)

latest_checkpoint = os.path.join(checkpoint_folder, "latest.keras")
best_checkpoint = os.path.join(checkpoint_folder, "best.keras")

latest_checkpoint_callback = ModelCheckpoint(
    latest_checkpoint, monitor="val_loss", save_best_only=False, save_weights_only=False, verbose=1
)

best_checkpoint_callback = ModelCheckpoint(
    best_checkpoint, monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=1
)

initial_epoch = 0
if os.path.exists(latest_checkpoint):
    model.load_weights(latest_checkpoint)
    with open(os.path.join(checkpoint_folder, "epoch.txt"), "rb") as f:
        initial_epoch = int(f.read())

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5000,
    initial_epoch=initial_epoch,
    batch_size=batch_size,
    callbacks=[tensorboard_callback, predictions_callback, latest_checkpoint_callback, best_checkpoint_callback,
               lr_scheduler],
    verbose=1
)
