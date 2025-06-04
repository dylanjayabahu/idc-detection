import random
import os
import numpy as np
from PIL import Image

from data_config import DATA_IMAGE_DIR
from parameters import *

def save_homogenous_oversamples(seed=DEFAULT_SEED):
    random.seed(seed)
    print("\nIdentifying Training IDs...")

    slide_ids = get_training_ids(DATA_IMAGE_DIR, seed)
    print(f"Training Patient IDs: {slide_ids}")

    total_positive, total_negative = count_patch_distribution(slide_ids, DATA_IMAGE_DIR)

    print("FOR TRAIN:")
    print(f"Total Positive Patches: {total_positive}")
    print(f"Total Negative Patches: {total_negative}")

    oversamples_needed = max(0, total_negative - total_positive)
    print(f"Positive Patches Needed for Balance: {oversamples_needed}")

    if oversamples_needed == 0:
        print("The dataset is already balanced. No oversampling required.")
        return

    homogeneous_regions = find_homogeneous_regions(slide_ids, DATA_IMAGE_DIR)
    print(f"Total Homogeneous Regions Available: {len(homogeneous_regions)}")

    if not homogeneous_regions:
        print("No homogeneous regions available. Exiting.")
        return

    regions_to_sample = select_regions_for_oversampling(homogeneous_regions, oversamples_needed)
    print(f"Generating {len(regions_to_sample)} new positive patches...")

    generate_oversampled_patches(regions_to_sample, DATA_IMAGE_DIR)
    print("Oversampling complete.")

def get_training_ids(data_dir, num_train=84, seed = DEFAULT_SEED):
    patient_ids = [
        int(folder) for folder in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, folder))
    ]
    random.seed(seed)
    random.shuffle(patient_ids)
    return patient_ids[:num_train]

def count_patch_distribution(slide_ids, data_dir):
    total_positive, total_negative = 0, 0
    for pid in slide_ids:
        for label in ['0', '1']:
            path = os.path.join(data_dir, str(pid), label)
            if os.path.isdir(path):
                count = len(os.listdir(path))
                if label == '1':
                    total_positive += count
                else:
                    total_negative += count
    return total_positive, total_negative

def find_homogeneous_regions(slide_ids, data_dir):
    regions = []
    for pid in slide_ids:
        positive_path = os.path.join(data_dir, str(pid), '1')
        if not os.path.isdir(positive_path):
            print(f"No positive folder found for Patient {pid}")
            continue

        positive_patches = {}
        for name in os.listdir(positive_path):
            if "class1" in name:
                try:
                    x = int(name.split('_x')[1].split('_y')[0])
                    y = int(name.split('_y')[1].split('_class')[0])
                    positive_patches[(x, y)] = name
                except Exception:
                    continue

        for (x, y) in positive_patches:
            if all((x + dx, y + dy) in positive_patches for dx, dy in [(0, 50), (50, 0), (50, 50)]):
                regions.append((pid, x, y))

        print(f"Patient {pid} - Homogeneous Regions Found: {len(regions)}")

    return regions

def select_regions_for_oversampling(homogeneous_regions, oversamples_needed):
    if len(homogeneous_regions) >= oversamples_needed:
        return random.sample(homogeneous_regions, oversamples_needed)
    else:
        extra = random.choices(homogeneous_regions, k=oversamples_needed - len(homogeneous_regions))
        return homogeneous_regions + extra

def generate_oversampled_patches(regions, data_dir):
    for pid, x, y in regions:
        base_path = os.path.join(data_dir, str(pid), '1')
        tl = np.array(Image.open(os.path.join(base_path, f'{pid}_idx5_x{x}_y{y}_class1.png')))
        tr = np.array(Image.open(os.path.join(base_path, f'{pid}_idx5_x{x+50}_y{y}_class1.png')))
        bl = np.array(Image.open(os.path.join(base_path, f'{pid}_idx5_x{x}_y{y+50}_class1.png')))
        br = np.array(Image.open(os.path.join(base_path, f'{pid}_idx5_x{x+50}_y{y+50}_class1.png')))
        
        full_img = np.vstack((np.hstack((tl, tr)), np.hstack((bl, br))))
        rand_x = random.randint(0, 50)
        rand_y = random.randint(0, 50)
        cropped = full_img[rand_y:rand_y+50, rand_x:rand_x+50]

        new_name = f'{pid}_x{x+rand_x}_y{y+rand_y}_class1.png'
        Image.fromarray(cropped).save(os.path.join(base_path, new_name))
