import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

from data_config import DATA_AUGMENTATIONS_DIR

def create_plain_dataset(x_data, y_data, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def create_augmented_dataset(x_data, y_data, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size)
    augmentation = augmentation_layer()
    dataset = dataset.map(lambda x, y: (augmentation(x), y))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset 

def save_sample_augmentations(augmentation_layer, sample_image, n=5, sample_id=0, save_dir=DATA_AUGMENTATIONS_DIR):
    """
    Save the original image and n augmented versions to disk for viewing.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    def normalize(img):
        if isinstance(img, np.ndarray):
            return img.astype(np.float32) / 255.0
        return tf.cast(img, tf.float32) / 255.0

    sample_image = normalize(sample_image)
    plt.imsave(os.path.join(save_dir, f"Original_{sample_id}.png"), (sample_image * 255).astype(np.uint8))

    for i in range(1, n + 1):
        augmented = augmentation_layer(tf.expand_dims(sample_image, 0))
        augmented_np = (augmented[0].numpy() * 255).astype(np.uint8)
        plt.imsave(os.path.join(save_dir, f"Augmentation_{sample_id}_{i}.png"), augmented_np)
    
    print(f"Saved {n} augmentations for sample {sample_id} in {save_dir}/.")

def custom_random_rotation(image):
    """
    Randomly rotate the image by 0/90/180/270 degrees
    """
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    return tf.image.rot90(image, k=k)

def augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"),  # Random horizontal and vertical flips
        tf.keras.layers.Lambda(lambda x: tf.map_fn(custom_random_rotation, x, dtype=tf.float32))
    ])