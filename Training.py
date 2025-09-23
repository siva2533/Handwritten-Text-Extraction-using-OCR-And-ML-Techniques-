import pandas as pd

import numpy as np

import cv2

import os

import datetime

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Bidirectional, LSTM, Dense, Reshape, Layer

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard



# --- Configuration ---

DATASET_BASE_PATH = "C:\\Users\\MAANISHA\\OneDrive\\Desktop\\tam\\datasets"

TRAIN_CSV = os.path.join(DATASET_BASE_PATH, "train", "train.csv")

VAL_CSV = os.path.join(DATASET_BASE_PATH, "val", "val.csv")

# TRAIN_IMAGE_DIR and VAL_IMAGE_DIR should point to the parent directory of 'train/' or 'val/'

# For example, if CSV has "train/1.jpg", and images are in "datasets/train/1.jpg",

# TRAIN_IMAGE_DIR should be "C:\Users\MAANISHA\OneDrive\Desktop\tamil\datasets"

TRAIN_IMAGE_DIR = DATASET_BASE_PATH

VAL_IMAGE_DIR = DATASET_BASE_PATH



IMG_HEIGHT = 32

IMG_WIDTH = 256

BATCH_SIZE = 32

EPOCHS = 30

MAX_TEXT_LEN = 70



# --- Character Set ---

def get_character_set(csv_paths):

    all_text = []

    for csv_path in csv_paths:

        if not os.path.exists(csv_path):

            print(f"Warning: CSV file not found at {csv_path}. Skipping.")

            continue

        df = pd.read_csv(csv_path)

        if 'text' in df.columns and not df['text'].empty:

            all_text.extend(df['text'].dropna().tolist())

    

    unique_chars = sorted(list(set("".join(all_text))))

    char_to_num = {char: i + 1 for i, char in enumerate(unique_chars)}

    char_to_num['[BLANK]'] = 0 

    

    num_to_char = {i + 1: char for i, char in enumerate(unique_chars)}

    num_to_char[0] = '[BLANK]'

    

    return char_to_num, num_to_char, len(unique_chars) + 1 



char_to_num, num_to_char, NUM_CLASSES = get_character_set([TRAIN_CSV, VAL_CSV])

print(f"Number of unique characters (including blank): {NUM_CLASSES}")

print(f"Character map sample: {list(char_to_num.items())[:10]}...")



# --- Data Generator/Dataset Class ---

class HTRDataset(keras.utils.Sequence):

    def __init__(self, df, image_dir, char_to_num, img_height, img_width, max_text_len, batch_size):

        self.df = df

        self.image_dir = image_dir

        self.char_to_num = char_to_num

        self.img_height = img_height

        self.img_width = img_width

        self.max_text_len = max_text_len

        self.batch_size = batch_size

        self.on_epoch_end()



    def __len__(self):

        return int(np.floor(len(self.df) / self.batch_size))



    def __getitem__(self, idx):

        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_df = self.df.iloc[batch_indices]



        batch_images = []

        batch_labels = []

        batch_input_length = []

        batch_label_length = []



        for _, row in batch_df.iterrows():

            # Dynamically determine the base image directory from the file_name itself

            # Assuming row['file_name'] is like "train/1.jpg" or "val/image.jpg"

            image_relative_path = row['file_name'].replace('\\', '/') # Ensure forward slashes for consistency

            

            img_path = os.path.join(self.image_dir, image_relative_path)

            

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)



            if image is None:

                print(f"[ERROR] Could not load image {img_path}. Skipping batch item.")

                continue 



            original_height, original_width = image.shape

            if original_height == 0 or original_width == 0:

                print(f"Warning: Image {img_path} has zero dimension. Skipping.")

                continue

            

            new_width = int(original_width * (self.img_height / original_height))

            image = cv2.resize(image, (new_width, self.img_height), interpolation=cv2.INTER_AREA)



            if new_width < self.img_width:

                pad_width = self.img_width - new_width

                image = np.pad(image, ((0, 0), (0, pad_width)), 'constant', constant_values=255)

            else:

                image = image[:, :self.img_width]



            image = image / 255.0

            image = np.expand_dims(image, axis=-1)



            text_label = str(row['text'])

            encoded_label = [self.char_to_num.get(char, self.char_to_num['[BLANK]']) for char in text_label]

            

            if len(encoded_label) > self.max_text_len:

                encoded_label = encoded_label[:self.max_text_len]

            label_length = len(encoded_label)



            padded_label = np.full(self.max_text_len, self.char_to_num['[BLANK]'], dtype=np.int32)

            padded_label[:len(encoded_label)] = encoded_label



            batch_images.append(image)

            batch_labels.append(padded_label)

            batch_input_length.append(self.img_width // 4) 

            batch_label_length.append(label_length)



        if not batch_images:

            return (

                tf.zeros((0, self.img_height, self.img_width, 1), dtype=tf.float32),

                tf.zeros((0, self.max_text_len), dtype=tf.int32),

                tf.zeros((0, 1), dtype=tf.int32),

                tf.zeros((0, 1), dtype=tf.int32)

            ), tf.zeros((0,), dtype=tf.float32)



        return (

            tf.convert_to_tensor(np.array(batch_images), dtype=tf.float32),

            tf.convert_to_tensor(np.array(batch_labels), dtype=tf.int32),

            tf.convert_to_tensor(np.array(batch_input_length), dtype=tf.int32),

            tf.convert_to_tensor(np.array(batch_label_length), dtype=tf.int32)

        ), tf.convert_to_tensor(np.zeros(len(batch_images)), dtype=tf.float32)



    def on_epoch_end(self):

        self.indices = np.arange(len(self.df))

        np.random.shuffle(self.indices)



    @property

    def output_signature(self):

        return (

            (tf.TensorSpec(shape=(None, self.img_height, self.img_width, 1), dtype=tf.float32),

             tf.TensorSpec(shape=(None, self.max_text_len), dtype=tf.int32),

             tf.TensorSpec(shape=(None, 1), dtype=tf.int32),

             tf.TensorSpec(shape=(None, 1), dtype=tf.int32)),

            tf.TensorSpec(shape=(None,), dtype=tf.float32)

        )



# --- Custom CTC Loss Layer ---

class CTCLossLayer(Layer):

    def __init__(self, name=None, **kwargs):

        super().__init__(name=name, **kwargs)

        self.loss_fn = tf.keras.backend.ctc_batch_cost



    def call(self, inputs):

        labels = tf.cast(inputs[0], tf.int32)

        predictions = inputs[1]

        input_length = tf.cast(inputs[2], tf.int32)

        label_length = tf.cast(inputs[3], tf.int32)



        loss = self.loss_fn(labels, predictions, input_length, label_length)

        self.add_loss(tf.reduce_mean(loss)) 

        return predictions



    def get_config(self):

        config = super().get_config()

        return config



# --- Model Definition (CRNN Architecture) ---

def build_crnn_model(input_shape, num_classes):

    input_img = keras.Input(shape=input_shape, name="image")

    labels = keras.Input(name="labels", shape=(None,), dtype="float32")

    input_length = keras.Input(name="input_length", shape=(1,), dtype="int64")

    label_length = keras.Input(name="label_length", shape=(1,), dtype="int64")



    # CNN Feature Extractor

    x = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="conv1")(input_img)

    x = MaxPooling2D((2, 2), name="pool1")(x) 

    x = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="conv2")(x)

    x = MaxPooling2D((2, 2), name="pool2")(x) 

    x = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="conv3")(x)

    

    # Pooling to reduce height to 1.

    x = MaxPooling2D((2, 1), name="pool3_height_reduction")(x) 

    x = MaxPooling2D((2, 1), name="pool4_height_reduction")(x) 

    x = MaxPooling2D((2, 1), name="pool5_height_reduction")(x) 

    

    # Reshape for RNN: (batch_size, width, features)

    shape = list(x.shape) 

    x = Reshape((shape[2], shape[1] * shape[3]))(x) 



    # Bidirectional LSTM layers

    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(x)

    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.25))(x)



    # Output layer with softmax activation for CTC

    output = Dense(num_classes, activation="softmax", name="dense_output")(x)



    loss_output = CTCLossLayer(name="ctc_loss")(

        [labels, output, input_length, label_length]

    )



    training_model = Model(inputs=[input_img, labels, input_length, label_length], outputs=loss_output)



    return training_model



# --- Main Training Script ---

if __name__ == "__main__":

    # --- Check and Print paths for debugging ---

    print(f"TRAIN_CSV: {TRAIN_CSV}")

    print(f"VAL_CSV: {VAL_CSV}")

    print(f"TRAIN_IMAGE_DIR (parent directory): {TRAIN_IMAGE_DIR}")

    print(f"VAL_IMAGE_DIR (parent directory): {VAL_IMAGE_DIR}")

    print("-" * 30)

    # ----------------------------------------



    try:

        train_df = pd.read_csv(TRAIN_CSV)

        val_df = pd.read_csv(VAL_CSV)

    except FileNotFoundError as e:

        print(f"Error loading CSV: {e}. Please ensure paths are correct and files exist.")

        print("Exiting training script.")

        exit()



    train_generator = HTRDataset(train_df, TRAIN_IMAGE_DIR, char_to_num,

                                 IMG_HEIGHT, IMG_WIDTH, MAX_TEXT_LEN, BATCH_SIZE)

    val_generator = HTRDataset(val_df, VAL_IMAGE_DIR, char_to_num,

                               IMG_HEIGHT, IMG_WIDTH, MAX_TEXT_LEN, BATCH_SIZE)



    model = build_crnn_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), num_classes=NUM_CLASSES)

    model.compile(optimizer=Adam(learning_rate=0.001)) 

    model.summary()



    model_checkpoint_callback = ModelCheckpoint(

        filepath="model.keras",

        save_best_only=True,

        monitor="val_loss",

        verbose=1

    )

    early_stopping_callback = EarlyStopping(

        monitor="val_loss",

        patience=10,

        restore_best_weights=True,

        verbose=1

    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)



    print("\n--- Starting Training ---")

    history = model.fit(

        train_generator,

        validation_data=val_generator,

        epochs=EPOCHS,

        callbacks=[model_checkpoint_callback, early_stopping_callback, tensorboard_callback],

        verbose=1

    )



    print("\n--- Training Complete! ---")

    print(f"Best model saved to htr_tamil_model_best.keras based on validation loss.")
