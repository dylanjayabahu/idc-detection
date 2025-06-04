import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random

from data_config import DATA_IMAGE_DIR, DATA_NP_DIR
from data_augmentation import *
from parameters import *
from homogenous_oversampling import *

def unison_shuffle_dataset(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def describe_data(a, b):
    print('Total number of images (x_data):', len(a))
    print('Total number of images (y_data):', len(a))
    print('Number of IDC(-) Images:', np.sum(b==0))
    print('Number of IDC(+) Images:', np.sum(b==1))
    print('Percentage of positive images:', str(round(100*np.mean(b), 2)) + '%') 
    print('Image shape (Width, Height, Channels):', a[0].shape)
    print()

def save_pngs_to_np_dataset(seed=DEFAULT_SEED, num_train = NUM_TRAIN, num_validation = NUM_VALIDATION, num_test = NUM_TEST):
    """
    Loads data from Janowczyk's paper containing PNG files, converts to NumPy arrays, 
    splits data into training, validation, and test sets as per the dataset split in the reference paper, 
    """
    random.seed(seed)
    
    print("\n\nLoading Images ... Will take some time ........")
    DATA_DIR = DATA_IMAGE_DIR
    slide_ids = {}

    for patient_id_folder in os.listdir(DATA_DIR):
        patient_path = os.path.join(DATA_DIR, patient_id_folder)

        if os.path.isdir(patient_path):
            patient_id = int(patient_id_folder)
            slide_ids[patient_id] = {'positive': [], 'negative': []}

            for class_folder in ['0', '1']:
                class_path = os.path.join(patient_path, class_folder)

                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        img_path = os.path.join(class_path, img_name)
                        try:
                            print(img_path)
                            img = Image.open(img_path).convert('RGB')  # Ensure image is RGB
                            img = img.resize((50, 50))  # Ensure image size is 50x50
                            img_array = np.array(img)

                            if img_array.shape == (50, 50, 3):
                                if class_folder == '1':
                                    slide_ids[patient_id]['positive'].append(img_array)
                                else:
                                    slide_ids[patient_id]['negative'].append(img_array)
                            else:
                                print(f"Skipping image with incorrect shape: {img_path}")

                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")

    # Collect patient IDs and shuffle
    all_patient_ids = list(slide_ids.keys())
    random.shuffle(all_patient_ids)

    # Split patient IDs into train, validation, and test
    train_ids = all_patient_ids[:num_train]
    validation_ids = all_patient_ids[num_train:num_train + num_validation]
    test_ids = all_patient_ids[num_train + num_validation:num_train + num_validation + num_test]


    def collect_images(patient_ids):
        positives, negatives = [], []
        for pid in patient_ids:
            positives.extend(slide_ids[pid]['positive'])
            negatives.extend(slide_ids[pid]['negative'])
        return np.array(positives), np.array(negatives)

    # Collect images for each set
    train_positives, train_negatives = collect_images(train_ids)
    val_positives, val_negatives = collect_images(validation_ids)
    test_positives, test_negatives = collect_images(test_ids)

    # Combine positives and negatives
    def combine_and_label(positives, negatives):
        x_data = np.concatenate([negatives, positives], axis=0)
        y_data = np.concatenate([np.zeros(len(negatives)), np.ones(len(positives))], axis=0)
        return x_data, y_data

    # # Oversample positives with augmentation to balance classes
    # num_negatives = len(train_negatives)
    # num_positives = len(train_positives)

    # if num_positives < num_negatives:
    #     num_to_generate = num_negatives - num_positives
    #     augmented_positives = augment_images(train_positives, num_to_generate)
    #     train_positives_augmented = np.concatenate([train_positives, augmented_positives], axis=0)
    #     y_train_positives_augmented = np.ones(len(train_positives_augmented))
    # else:
    #     train_positives_augmented = train_positives
    #     y_train_positives_augmented = np.ones(len(train_positives))

    # Combine oversampled positives and negatives
    # x_train_oversampled = np.concatenate([train_negatives, train_positives_augmented], axis=0)
    # y_train_oversampled = np.concatenate([np.zeros(len(train_negatives)), y_train_positives_augmented], axis=0)


    # Shuffle datasets
    x_train_oversampled, y_train_oversampled = unison_shuffle_dataset(*combine_and_label(train_positives, train_negatives))
    x_validation, y_validation = unison_shuffle_dataset(*combine_and_label(val_positives, val_negatives))
    x_test, y_test = unison_shuffle_dataset(*combine_and_label(test_positives, test_negatives))

    # Save datasets
    np.save(os.path.join(DATA_NP_DIR, 'X_TRAIN.npy'), x_train_oversampled)
    np.save(os.path.join(DATA_NP_DIR, 'Y_TRAIN.npy'), y_train_oversampled)
    np.save(os.path.join(DATA_NP_DIR, 'X_VALIDATION.npy'), x_validation)
    np.save(os.path.join(DATA_NP_DIR, 'Y_VALIDATION.npy'), y_validation)
    np.save(os.path.join(DATA_NP_DIR, 'X_TEST.npy'), x_test)
    np.save(os.path.join(DATA_NP_DIR, 'Y_TEST.npy'), y_test)

    print(f"Training set: {len(train_ids)} slides, Validation set: {len(validation_ids)} slides, Test set: {len(test_ids)} slides.")
    
    # Print positive, negative, and total patch counts for each set
    print(f"Train: {len(train_positives)} positive, {len(train_negatives)} negative, {len(train_positives) + len(train_negatives)} total patches.")
    print(f"Validation: {len(val_positives)} positive, {len(val_negatives)} negative, {len(val_positives) + len(val_negatives)} total patches.")
    print(f"Test: {len(test_positives)} positive, {len(test_negatives)} negative, {len(test_positives) + len(test_negatives)} total patches.")

    return

def load_data(batch_size=BATCH_SIZE, show_data_stats=True, augment_data=False, num_aug_visualizations=0):
    """
    Loads the training, validation, and test datasets.
    Applies normalization, (optional) data augmentation, and returns datasets for training and evaluation.
    """
    
    ########################################## LOAD FILES ##########################################
    print("Loading Files ...")
    
    x_test = np.load(os.path.join(DATA_NP_DIR, 'X_TEST.npy'))  # Test images
    y_test = np.load(os.path.join(DATA_NP_DIR, 'Y_TEST.npy'))  # Test labels

    x_validation = np.load(os.path.join(DATA_NP_DIR, 'X_VALIDATION.npy'))  # Validation images
    y_validation = np.load(os.path.join(DATA_NP_DIR, 'Y_VALIDATION.npy'))  # Validation labels

    x_train = np.load(os.path.join(DATA_NP_DIR, 'X_TRAIN.npy'))  # Oversampled training images
    y_train = np.load(os.path.join(DATA_NP_DIR, 'Y_TRAIN.npy'))  # Oversampled training labels

    print("Finished Loading Files")     

    ######################################### DESCRIBE DATA #########################################
    # print("TOTAL")
    # print(f"Total Patches: {len(y_test) + len(y_validation) + len(y_train)}")
    # print(f"Total Positive Patches: {sum(y_test) + sum(y_validation) + sum(y_train)}")
    # print()

    # print(f"TRAIN")
    # print(f"Total Patches: {len(y_train)}")
    # print(f"Total Positiev Patches: {sum(y_train)}")

    ########################################## NORMALIZE DATA ##########################################
    print("Normalizing Data...")
    x_test = x_test.astype('float32') / 255.0
    x_validation = x_validation.astype('float32') / 255.0
    x_train = x_train.astype('float32') / 255.0
    print("Finished Normalizing Data")

    ################################## DESCRIBE & ONE-HOT ENCODE #########################################
    # Describe the data to ensure everything is loaded correctly
    if show_data_stats:
        print('\n', '-'*10 + 'TRAINING DATA' + '-'*10)
        describe_data(x_train, y_train)

        print('\n', '-'*10 + 'VALIDATION DATA' + '-'*10)
        describe_data(x_validation, y_validation)

        print('\n', '-'*10 + 'TESTING DATA' + '-'*10)
        describe_data(x_test, y_test)

    # One-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train)
    y_validation = tf.keras.utils.to_categorical(y_validation)
    y_test = tf.keras.utils.to_categorical(y_test)

    ########################################## AUGMENT DATA ##########################################
    # Apply data augmentation if requested
    if augment_data:
        train_ds = create_augmented_dataset(x_train, y_train, batch_size=batch_size)
        for i in range(num_aug_visualizations):
            save_sample_augmentations(augmentation_layer=augmentation_layer(), sample_image=x_train[i], n=8, id=i)
    else:
        train_ds = create_plain_dataset(x_train, y_train, batch_size=batch_size)

    validation_ds = create_plain_dataset(x_validation, y_validation, batch_size=batch_size)
    
    ########################################## RETURN ###############################################
    print('\n' + '-'*15 + 'DATA LOADED SUCCESSFULLY' + '-'*15 + "\n\n")

    return train_ds, validation_ds, x_test, y_test