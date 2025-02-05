import tensorflow as tf
from tensorflow.keras.layers import (Lambda, Reshape, Permute, Input, GaussianNoise, concatenate, Conv2D,
                                     ConvLSTM2D, BatchNormalization, TimeDistributed, Add, Dropout, 
                                     MaxPooling2D, UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
import wandb
import numpy as np
import glob
import matplotlib.pyplot as plt
from keras.preprocessing import image

# Hyperparameters
IMG_HEIGHT = 256
IMG_WIDTH = 444
FRAME_COUNT = 5
BATCH_SIZE = 4
EPOCHS = 25

# Initialize Weights & Biases
wandb.init(config={"num_epochs": EPOCHS, "batch_size": BATCH_SIZE, "height": IMG_HEIGHT, "width": IMG_WIDTH})
config = wandb.config

# Data Loading Function
def load_data_sequence(img_folder, img_width, img_height, sequence_length=5):
    img_paths = sorted(glob.glob(f"{img_folder}/*.png"))
    num_sequences = len(img_paths) - sequence_length + 1
    
    video_data, next_frame_data = [], []
    for seq_start in range(num_sequences):
        seq_imgs = [image.img_to_array(image.load_img(img_paths[i], target_size=(img_width, img_height), color_mode='rgb')) 
                    for i in range(seq_start, seq_start + sequence_length)]
        seq_imgs = [(img - 127.5) / 127.5 for img in seq_imgs]
        video_data.append(np.concatenate(seq_imgs[:-1], axis=-1))
        next_frame_data.append(seq_imgs[-1])
    
    return np.array(video_data), np.array(next_frame_data)

# Load Dataset
dataset_path = '/content/drive/MyDrive/New-Path/Data/Project'
videos, next_frames = load_data_sequence(dataset_path, IMG_WIDTH, IMG_HEIGHT, FRAME_COUNT)

# Train-Validation Split
train_X, val_X, train_y, val_y = train_test_split(videos, next_frames, test_size=0.2, random_state=42, shuffle=False)

# Custom Loss Function
def perceptual_distance(y_true, y_pred):
    y_true, y_pred = y_true * 255., y_pred * 255.
    rmean = (y_true[..., 0] + y_pred[..., 0]) / 2
    r, g, b = y_true[..., 0] - y_pred[..., 0], y_true[..., 1] - y_pred[..., 1], y_true[..., 2] - y_pred[..., 2]
    return K.mean(K.sqrt((((512 + rmean) * r * r) / 256) + 4 * g * g + (((767 - rmean) * b * b) / 256)))

# Model Definition
def build_model(input_shape, filters=4):
    inp = Input(shape=input_shape)
    reshaped = Reshape((input_shape[0], input_shape[1], 4, 3))(inp)
    permuted = Permute((1, 2, 4, 3))(reshaped)
    noise = GaussianNoise(0.1)(permuted)
    last_layer = Lambda(lambda x: x[:, :, :, :, -1], output_shape=(input_shape[0], input_shape[1], 3))(noise)
    
    x = ConvLSTM2D(filters=filters, kernel_size=(3,3), padding='same', return_sequences=True)(noise)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(x)
    
    x = ConvLSTM2D(filters=filters*2, kernel_size=(3,3), padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(x)
    
    x = ConvLSTM2D(filters=filters*4, kernel_size=(3,3), padding='same', return_sequences=True)(x)
    x = TimeDistributed(UpSampling2D(size=(2, 2)))(x)
    x = ConvLSTM2D(filters=filters*4, kernel_size=(3,3), padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    
    x = ConvLSTM2D(filters=filters*2, kernel_size=(3,3), padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Add()([x, last_layer])
    x = Dropout(0.2)(x)
    x = TimeDistributed(UpSampling2D(size=(2, 2)))(x)
    
    x = ConvLSTM2D(filters=filters, kernel_size=(3,3), padding='same', return_sequences=False)(x)
    x = BatchNormalization()(x)
    combined = concatenate([last_layer, x])
    output = Conv2D(3, (1,1))(combined)
    
    model = Model(inputs=inp, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=[perceptual_distance])
    return model

# Train Model
model = build_model((IMG_HEIGHT, IMG_WIDTH, FRAME_COUNT * 3))
model.fit(train_X, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(val_X, val_y))

# Save & Load Model
model.save('/content/drive/MyDrive/Model.h5')
model = tf.keras.models.load_model('/content/drive/MyDrive/Model.h5', custom_objects={'perceptual_distance': perceptual_distance})