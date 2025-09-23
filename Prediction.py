import pandas as pd

import numpy as np

import cv2

import os

import editdistance

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

import seaborn as sns

from collections import Counter

import random

import datetime



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import Layer



# --- Configuration ---

DATASET_BASE_PATH = "C:\\Users\\MAANISHA\\OneDrive\\Desktop\\tamil\\datasets"

TEST_CSV = os.path.join(DATASET_BASE_PATH, "test", "test.csv")

TEST_IMAGE_DIR = DATASET_BASE_PATH

MODEL_PATH = "model.keras" # Using 'model.keras' as requested



IMG_HEIGHT = 32

IMG_WIDTH = 256

MAX_TEXT_LEN = 50



# --- Output Folder for Plots ---

# Create a unique folder for each run's plots based on timestamp

PLOT_OUTPUT_DIR = os.path.join("evaluation_plots", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

print(f"Saving plots to: {os.path.abspath(PLOT_OUTPUT_DIR)}")



# --- Matplotlib Font Configuration for Tamil (using Noto Sans Tamil) ---

# **UPDATED PATH BASED ON YOUR NEW INPUT**

TAMIL_FONT_PATH = r"C:\Users\MAANISHA\OneDrive\Desktop\tam\NotoSansTamil-VariableFont_wdth,wght.ttf"



# Variable to hold the FontProperties object for the Tamil font

tamil_font_prop = None



# Check if the font file exists and configure Matplotlib

if not os.path.exists(TAMIL_FONT_PATH):

    print(f"CRITICAL ERROR: Tamil font file not found at {TAMIL_FONT_PATH}")

    print("Please ensure the path is correct and the font is installed.")

    # Fallback to a generic serif font if font not found

    plt.rcParams['font.family'] = 'serif'

else:

    try:

        # Create a FontProperties object directly from the file path

        tamil_font_prop = fm.FontProperties(fname=TAMIL_FONT_PATH)

        print(f"Matplotlib configured to use Tamil font: '{tamil_font_prop.get_name()}' from {TAMIL_FONT_PATH}")



        # Ensure unicode minus is handled correctly for numbers (important for plots)

        plt.rcParams['axes.unicode_minus'] = False



        # Set the Tamil font as the primary font for ALL generic font families.

        # This makes it the first choice regardless of what generic family Matplotlib requests.

        plt.rcParams['font.sans-serif'] = [tamil_font_prop.get_name()] + plt.rcParams['font.sans-serif']

        plt.rcParams['font.serif'] = [tamil_font_prop.get_name()] + plt.rcParams['font.serif']

        plt.rcParams['font.monospace'] = [tamil_font_prop.get_name()] + plt.rcParams['font.monospace']



        # Finally, set the default font family to 'sans-serif', which now prioritizes the Tamil font.

        plt.rcParams['font.family'] = 'sans-serif'



    except Exception as e:

        print(f"Error configuring Matplotlib font using path '{TAMIL_FONT_PATH}': {e}")

        print("Plots may not display Tamil characters correctly. Falling back to default system font.")

        tamil_font_prop = None # Ensure it's None on error

        plt.rcParams['font.family'] = 'serif'



# --- Custom CTC Loss Layer (Must be defined here too for for loading model!) ---

# This custom layer is essential for loading models that use CTC loss.

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



# --- Character Set ---

# Generates the character-to-number and number-to-character mappings

# based on text data from train, val, and test CSVs.

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

    char_to_num['[BLANK]'] = 0 # CTC requires a blank token



    num_to_char = {i + 1: char for i, char in enumerate(unique_chars)}

    num_to_char[0] = '[BLANK]'



    return char_to_num, num_to_char, len(unique_chars) + 1 # +1 for blank token



# Load character set using all relevant CSVs

char_to_num, num_to_char, NUM_CLASSES = get_character_set([

    os.path.join(DATASET_BASE_PATH, "train", "train.csv"),

    os.path.join(DATASET_BASE_PATH, "val", "val.csv"),

    TEST_CSV

])

print(f"Loaded Character map sample: {list(num_to_char.items())[:10]}...")



# --- Image Preprocessing Function ---

# Preprocesses a single image for model input and optionally returns

# the original image for plotting.

def preprocess_image(image_path_from_df, img_height, img_width, base_image_dir, for_display=False):

    full_img_path = os.path.join(base_image_dir, image_path_from_df.replace('\\', '/'))



    image = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)

    if image is None:

        print(f"Warning: Could not read image at {full_img_path}. Skipping.")

        return None, None if for_display else None



    original_image_for_display = image.copy() if for_display else None



    original_height, original_width = image.shape

    if original_height == 0 or original_width == 0:

        print(f"Warning: Empty image found at {full_img_path}. Skipping.")

        return None, None if for_display else None



    new_width = int(original_width * (img_height / original_height))

    image = cv2.resize(image, (new_width, img_height), interpolation=cv2.INTER_AREA)



    if new_width < img_width:

        pad_width = img_width - new_width

        image = np.pad(image, ((0, 0), (0, pad_width)), 'constant', constant_values=255) # Pad with white

    else:

        image = image[:, :img_width] # Crop if too wide



    image_normalized = image / 255.0  # Normalize pixel values to [0, 1]

    image_expanded = np.expand_dims(image_normalized, axis=0)  # Add batch dimension

    image_final = np.expand_dims(image_expanded, axis=-1) # Add channel dimension (for grayscale)



    return (image_final, original_image_for_display) if for_display else image_final



# --- CTC Decoder ---

# Decodes the raw predictions from the CTC model into readable text.

def decode_batch_predictions(pred_output, num_to_char_map):

    # The length of the input sequences to the CTC decoder

    input_len = np.full(pred_output.shape[0], pred_output.shape[1])

    # Use greedy decoding to get the most probable sequence

    results = tf.keras.backend.ctc_decode(pred_output, input_length=input_len, greedy=True)[0][0]



    output_text = []

    for res in results.numpy():

        # Map numerical predictions back to characters, ignoring blank and -1 tokens

        decoded = "".join([num_to_char_map.get(char_id, '') for char_id in res if char_id != -1 and char_id != 0])

        output_text.append(decoded.strip())

    return output_text



# --- Main Prediction and Evaluation Script ---

if __name__ == "__main__":

    try:

        # Load the trained model. compile=False is used because we only need inference.

        loaded_model = load_model(MODEL_PATH, compile=False, custom_objects={'CTCLossLayer': CTCLossLayer})

        # Extract the prediction output layer for inference

        prediction_output_layer = loaded_model.get_layer('dense_output').output

        inference_model = Model(inputs=loaded_model.inputs[0], outputs=prediction_output_layer)

        print("Model loaded successfully for prediction.")

        inference_model.summary()



    except Exception as e:

        print(f"Error loading model from {MODEL_PATH}: {e}")

        print("Ensure the model path is correct and the model was saved properly.")

        print("Exiting prediction script.")

        exit()



    try:

        test_df = pd.read_csv(TEST_CSV)

    except FileNotFoundError as e:

        print(f"Error loading test CSV: {e}. Please ensure path is correct.")

        print("Exiting prediction script.")

        exit()



    all_ground_truths = []

    all_predictions = []

    per_image_results = []

    images_for_display = [] # To store image data for plotting samples



    print("\n--- Starting Predictions on Test Set ---")

    for index, row in test_df.iterrows():

        img_filename_from_df = row['file_name']

        ground_truth_text = str(row['text']).strip() # Ensure text is string and stripped



        # Initialize original_image_for_display to None at the start of each loop

        # This prevents NameError if the assignment below somehow fails or is skipped.

        original_image_for_display = None



        # Preprocess image for model input and optionally get original for display

        preprocessed_img_model_input, original_image_for_display = preprocess_image(

            img_filename_from_df, IMG_HEIGHT, IMG_WIDTH, TEST_IMAGE_DIR, for_display=True

        )



        if preprocessed_img_model_input is None:

            continue # Skip if image couldn't be processed



        # Get raw predictions from the inference model

        raw_predictions = inference_model.predict(preprocessed_img_model_input, verbose=0)

        # Decode raw predictions into human-readable text

        predicted_texts = decode_batch_predictions(raw_predictions, num_to_char)

        predicted_text = predicted_texts[0] # Take the first (and only) prediction from the batch



        all_ground_truths.append(ground_truth_text)

        all_predictions.append(predicted_text)



        # Calculate Character Error Rate for the current image

        cer_single_image = editdistance.eval(ground_truth_text, predicted_text) / len(ground_truth_text) if len(ground_truth_text)>0 else 0



        # Store results for per-image analysis and overall metrics

        per_image_results.append({

            'file_name': img_filename_from_df,

            'ground_truth': ground_truth_text,

            'prediction': predicted_text,

            'cer': cer_single_image

        })



        # Store image data for plotting correct/incorrect samples later

        if original_image_for_display is not None:

             images_for_display.append({

                'image_data': original_image_for_display,

                'ground_truth': ground_truth_text,

                'prediction': predicted_text,

                'cer': cer_single_image

            })



        # Print progress every 100 images

        if (index + 1) % 100 == 0:

            print(f"Processed {index + 1}/{len(test_df)} images.")

            print(f"  Example: GT='{ground_truth_text}', Pred='{predicted_text}', CER={cer_single_image:.4f}")



    print("\n--- Evaluation Metrics ---")



    # Calculate Overall Character Error Rate (CER)

    total_char_distance = 0

    total_ground_truth_chars = 0

    for gt, pred in zip(all_ground_truths, all_predictions):

        total_char_distance += editdistance.eval(gt, pred)

        total_ground_truth_chars += len(gt)



    overall_cer = total_char_distance / total_ground_truth_chars if total_ground_truth_chars > 0 else 0

    print(f"Overall Character Error Rate (CER): {overall_cer:.4f}")

    # Calculate and print Character Accuracy

    character_accuracy = (1 - overall_cer) * 100

    print(f"Overall Character Accuracy: {character_accuracy:.2f}%")



    # Calculate Overall Word Error Rate (WER)

    total_word_distance = 0

    total_ground_truth_words = 0

    for gt, pred in zip(all_ground_truths, all_predictions):

        gt_words = gt.split()

        pred_words = pred.split()

        total_word_distance += editdistance.eval(gt_words, pred_words)

        total_ground_truth_words += len(gt_words)



    overall_wer = total_word_distance / total_ground_truth_words if total_ground_truth_words > 0 else 0

    print(f"Overall Word Error Rate (WER): {overall_wer:.4f}")

    # Calculate and print Word Accuracy

    word_accuracy = (1 - overall_wer) * 100

    print(f"Overall Word Accuracy: {word_accuracy:.2f}%")



    print("\n--- Per-Image Results Sample (First 10) ---")

    results_df = pd.DataFrame(per_image_results)

    print(results_df.head(10).to_string())



    ## Plotting and Saving Results



    ### Character-level Confusion Matrix

    print("\n--- Plotting Character-level Confusion Matrix ---")

    flat_ground_truths_chars = [char for text in all_ground_truths for char in text]

    flat_predictions_chars = [char for text in all_predictions for char in text]



    min_len_cm = min(len(flat_ground_truths_chars), len(flat_predictions_chars))

    flat_ground_truths_chars = flat_ground_truths_chars[:min_len_cm]

    flat_predictions_chars = flat_predictions_chars[:min_len_cm]



    # Identify characters involved in errors and common characters

    error_chars = set()

    for gt_char, pred_char in zip(flat_ground_truths_chars, flat_predictions_chars):

        if gt_char != pred_char:

            error_chars.add(gt_char)

            error_chars.add(pred_char)



    # Get the top 20 most common characters

    common_chars = [char for char, count in Counter(flat_ground_truths_chars).most_common(20)]

    # Combine common characters and error-involved characters for plotting

    chars_to_plot = sorted(list(set(common_chars).union(error_chars)))

    if '[BLANK]' in chars_to_plot:

        chars_to_plot.remove('[BLANK]') # Remove blank token if present



    if len(chars_to_plot)<2:

        print("Cannot generate confusion matrix plot: Not enough unique characters to compare or only one character type to plot.")

    else:

        # Create a mapping for characters to integers for the confusion matrix

        cm_char_to_int = {char: i for i, char in enumerate(chars_to_plot)}



        # Convert ground truth and prediction character lists to integer lists

        y_true_int = [cm_char_to_int.get(char, -1) for char in flat_ground_truths_chars]

        y_pred_int = [cm_char_to_int.get(char, -1) for char in flat_predictions_chars]



        # Filter out -1 values (characters not in our limited set for plotting)

        y_true_int = [val for val in y_true_int if val != -1]

        y_pred_int = [val for val in y_pred_int if val != -1]



        if len(y_true_int) > 0 and len(y_pred_int)>0:

            # Calculate the confusion matrix

            cm = confusion_matrix(y_true_int, y_pred_int, labels=list(range(len(chars_to_plot))))



            # Determine figure size dynamically for clarity

            fig_width = max(12, len(chars_to_plot) * 0.6) # Increased minimum width

            fig_height = max(12, len(chars_to_plot) * 0.6) # Increased minimum height

            plt.figure(figsize=(fig_width, fig_height))



            # Create mapping back from int to char for axis labels

            cm_int_to_char = {i: char for char, i in cm_char_to_int.items()}



            # Plot heatmap

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'shrink': 0.8},

                        xticklabels=[cm_int_to_char.get(i, '') for i in range(len(chars_to_plot))],

                        yticklabels=[cm_int_to_char.get(i, '') for i in range(len(chars_to_plot))])



            # Apply font properties directly to text elements if tamil_font_prop is defined

            plt.xlabel('Predicted Character', fontsize=14, fontproperties=tamil_font_prop if tamil_font_prop else None)

            plt.ylabel('True Character', fontsize=14, fontproperties=tamil_font_prop if tamil_font_prop else None)

            plt.title('Character-level Confusion Matrix (Selected Characters)', fontsize=16, fontproperties=tamil_font_prop if tamil_font_prop else None)



            # Apply font to tick labels

            if tamil_font_prop:

                plt.xticks(rotation=90, fontsize=12, fontproperties=tamil_font_prop)

                plt.yticks(rotation=0, fontsize=12, fontproperties=tamil_font_prop)

            else:

                plt.xticks(rotation=90, fontsize=12) # Fallback without custom font

                plt.yticks(rotation=0, fontsize=12) # Fallback without custom font



            plt.tight_layout() # Adjust layout to prevent labels from overlapping



            # Save plot

            cm_plot_path = os.path.join(PLOT_OUTPUT_DIR, "confusion_matrix.png")

            plt.savefig(cm_plot_path, bbox_inches='tight', dpi=300) # High DPI for clarity

            print(f"Confusion Matrix plot saved to: {cm_plot_path}")

            plt.close() # Close the plot to free memory

        else:

            print("Cannot generate confusion matrix plot: Filtered character lists for plotting are empty.")



    ### CER Distribution

    print("\n--- Plotting CER Distribution ---")

    cers = results_df['cer'].tolist()

    if cers:

        plt.figure(figsize=(10, 6))

        sns.histplot(cers, bins=20, kde=True)

        # Apply font properties directly

        plt.title('Distribution of Character Error Rates (CER) Per Image', fontsize=14, fontproperties=tamil_font_prop if tamil_font_prop else None)

        plt.xlabel('Character Error Rate (CER)', fontsize=12, fontproperties=tamil_font_prop if tamil_font_prop else None)

        plt.ylabel('Number of Images', fontsize=12, fontproperties=tamil_font_prop if tamil_font_prop else None)

        plt.grid(axis='y', alpha=0.75)



        # Apply font to tick labels

        if tamil_font_prop:

            plt.xticks(fontproperties=tamil_font_prop)

            plt.yticks(fontproperties=tamil_font_prop)



        # Save plot

        cer_dist_plot_path = os.path.join(PLOT_OUTPUT_DIR, "cer_distribution.png")

        plt.savefig(cer_dist_plot_path, bbox_inches='tight', dpi=300)

        print(f"CER Distribution plot saved to: {cer_dist_plot_path}")

        plt.close()

    else:

        print("No CER data to plot distribution.")



    ### Sample Predictions (Correct & Incorrect)

    print("\n--- Plotting Sample Predictions (Correct&Incorrect) ---")

    if images_for_display:

        correct_predictions = [item for item in images_for_display if item['cer'] == 0]

        incorrect_predictions = [item for item in images_for_display if item['cer']>0]



        # Select a few correct and incorrect samples to display

        num_samples_to_plot = min(5, len(correct_predictions))

        num_incorrect_samples_to_plot = min(5, len(incorrect_predictions))



        selected_samples = []

        if num_samples_to_plot > 0:

            selected_samples.extend(random.sample(correct_predictions, num_samples_to_plot))

        if num_incorrect_samples_to_plot > 0:

            selected_samples.extend(random.sample(incorrect_predictions, num_incorrect_samples_to_plot))



        if not selected_samples:

            print("No samples available to plot.")

        else:

            # Sort samples by CER (highest error first) for better visualization

            selected_samples.sort(key=lambda x: x['cer'], reverse=True)



            num_plots = len(selected_samples)

            fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots)) # Adjust figure height based on number of samples



            if num_plots == 1:

                axes = [axes] # Ensure axes is iterable even for a single plot



            for i, sample in enumerate(selected_samples):

                ax = axes[i]

                image_to_show = sample['image_data']



       
