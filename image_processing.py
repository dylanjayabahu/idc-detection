import cv2
import numpy as np
import os
from PIL import Image

from data_config import *
from parameters import *

def save_slide_image(dir, np_image, name):
    im = Image.fromarray(np_image)
    output_path = os.path.join(dir, name)
    im.save(output_path)

def stitch_slide_patches(patient_id):
    """
    Stiches the entire histology slide for a given patient ID by combining all image patches.
    """
    patient_folder = os.path.join(DATA_IMAGE_DIR, patient_id)

    # Initialize lists to store patches and their coordinates
    patches = []
    x_coords = []
    y_coords = []
    
    # Iterate through both IDC negative (0) and IDC positive (1) folders
    for class_label in ['0', '1']:
        class_folder = os.path.join(patient_folder, class_label)
        for filename in os.listdir(class_folder):
            if filename.endswith(".png"):
                print("   "*50, end='\r')
                print("Patching Filename " + filename + " ...", end='\r')

                # Parse the filename to extract coordinates
                parts = filename.split('_')
                x = int(parts[2][1:])  # Extract X coordinate (e.g., x1351 -> 1351)
                y = int(parts[3][1:])  # Extract Y coordinate (e.g., y1101 -> 1101)
                
                # Load the image patch
                img = Image.open(os.path.join(class_folder, filename))
                img = np.array(img)
                
                # Pad the image if it's smaller than PATCH_SIZExPATCH_SIZE (50x50)
                if img.shape != (PATCH_SIZE, PATCH_SIZE, 3):
                    img = np.pad(img, 
                                 ((0, PATCH_SIZE - img.shape[0]), (0, PATCH_SIZE - img.shape[1]), (0, 0)), 
                                 mode='constant', constant_values=0)
                
                # Store the image patch and its coordinates
                patches.append(img)
                x_coords.append(x)
                y_coords.append(y)
    
    # Calculate the size of the full slide
    max_x = max(x_coords) + PATCH_SIZE  # Add 50 to get the rightmost boundary
    max_y = max(y_coords) + PATCH_SIZE  # Add 50 to get the bottom boundary
    
    # Initialize an empty canvas for the full slide
    full_slide = np.ones((max_y, max_x, 3), dtype=np.uint8) * 255

    # Place each patch on the canvas
    for patch, x, y in zip(patches, x_coords, y_coords):
        full_slide[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :] = patch

    return full_slide

def stitch_and_apply_patched_heatmap(patient_id, model, heatmap_weighting = HEATMAP_WEIGHTING, blurring_kernel_size = 41, blurring_sigma = 100):
    """
    Stitches slide and applies a heatmap to the stitched slide for a given patient using the model's predictions.
    Apples model prediction on each patch in the image
    """
    patient_folder = os.path.join(DATA_IMAGE_DIR, str(patient_id))

    # Initialize lists to store patches and their coordinates
    patches = []
    x_coords = []
    y_coords = []
    predictions = []
    
    # Iterate through both IDC negative (0) and IDC positive (1) folders
    for class_label in ['0', '1']:
        class_folder = os.path.join(patient_folder, class_label)
        for filename in os.listdir(class_folder):
            if filename.endswith(".png"):
                print("   "*50, end='\r')
                print("Patching Filename " + filename + " ...", end='\r')
                # Parse the filename to extract coordinates
                parts = filename.split('_')
                x = int(parts[2][1:])  # Extract X coordinate (e.g., x1351 -> 1351)
                y = int(parts[3][1:])  # Extract Y coordinate (e.g., y1101 -> 1101)
                
                # Load the image patch and preprocess for the model
                img = Image.open(os.path.join(class_folder, filename))
                img = np.array(img)
                
                # Pad the image if it's smaller than 50x50
                if img.shape != (PATCH_SIZE, PATCH_SIZE, 3):
                    img = np.pad(img, 
                                 ((0, PATCH_SIZE - img.shape[0]), (0, PATCH_SIZE - img.shape[1]), (0, 0)), 
                                 mode='constant', constant_values=0)
                
                # Preprocess for the model and get prediction
                img_preprocessed = np.expand_dims(img, axis=0) / 255.0
                prediction = model.predict(img_preprocessed, verbose=None)
                
                # Store the prediction, image patch, and coordinates
                patches.append(img)
                x_coords.append(x)
                y_coords.append(y)
                predictions.append(prediction[0][1])  # Confidence for IDC positive (class 1)
    
    # Calculate the size of the full slide
    max_x = max(x_coords) + PATCH_SIZE  # Add 50 to get the rightmost boundary
    max_y = max(y_coords) + PATCH_SIZE  # Add 50 to get the bottom boundary
    
    # Initialize an empty canvas for the full slide
    full_slide = np.ones((max_y, max_x, 3), dtype=np.uint8) * 255
    heatmap_overlay = np.zeros((max_y, max_x), dtype=np.float32)

    # Place each patch on the canvas and add prediction values to the heatmap
    for patch, x, y, pred in zip(patches, x_coords, y_coords, predictions):
        full_slide[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :] = patch
        heatmap_overlay[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = pred

    # Normalize the heatmap overlay to range [0, 1]
    min_val = np.min(heatmap_overlay)
    max_val = np.max(heatmap_overlay)
    range_val = max_val - min_val
    if range_val == 0:
        heatmap_overlay = np.zeros_like(heatmap_overlay)
    else:
        heatmap_overlay = (heatmap_overlay - min_val) / range_val


    # Apply a colormap to the heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap_overlay), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper overlay
    if blurring_kernel_size%2 == 0:
        blurring_kernel_size += 1
    smooth_heatmap = cv2.GaussianBlur(heatmap, (blurring_kernel_size, blurring_kernel_size), blurring_sigma)

    # Overlay the heatmap on the original slide
    overlayed_image = cv2.addWeighted(full_slide, 1-heatmap_weighting, heatmap, heatmap_weighting, 0)
    overlayed_image_smooth = cv2.addWeighted(full_slide, 1-heatmap_weighting, smooth_heatmap, heatmap_weighting, 0)
    
    return full_slide, heatmap, smooth_heatmap, overlayed_image, overlayed_image_smooth

def overlay_threshold_heatmap(wsi_image, heatmap, alpha=OVERLAY_THRESHOLD):
    """
    Overlay the heatmap on top of the whole slide image.
    If the heatmap prediction at a pixel is below the alpha threshold, show the WSI image.
    If the heatmap prediction is above the alpha threshold, show the heatmap with no transparency.
    """
    # Ensure heatmap is in the range 0 to 1
    heatmap_normalized = np.clip(heatmap, 0, 1)
    
    # Apply a color map to the heatmap for better visualization (e.g., 'jet' colormap)
    heatmap_color = cv2.applyColorMap((255 * (1 - heatmap_normalized)).astype(np.uint8), cv2.COLORMAP_JET)

    # Create a mask where heatmap values are above the alpha threshold
    mask = heatmap_normalized >= alpha
    
    # Initialize the result with the WSI image
    overlayed_image = np.copy(wsi_image)
    
    # For pixels where the heatmap value is above alpha, show the heatmap instead of the WSI
    overlayed_image[mask] = heatmap_color[mask]
    
    return overlayed_image

def stitch_and_apply_sliding_heatmap(patient_id, model, batch_size=SLIDING_WINDOW_BATCH_SIZE, step_size = SLIDING_WINDOW_STEP_SIZE):
    """
    Stitches slide and applies a heatmap to the stitched slide for a given patient using the model's predictions.
    Apples model prediction with a sliding window using he specified step size
    """

    wsi_image = stitch_slide_patches(str(patient_id))
    window_size = PATCH_SIZE

    print(f"Sliding window predictions with steps of {step_size} on image of dims ({wsi_image.shape[0]},{wsi_image.shape[1]}), patient_id = {patient_id}")
    
    # Initialize heatmap with zeros, the same size as the WSI
    heatmap = np.zeros((wsi_image.shape[0], wsi_image.shape[1]), dtype=np.float32)
    
    # Store the number of predictions made at each pixel (for averaging)
    prediction_count = np.zeros_like(heatmap)

    step_size = min(window_size, step_size)  # Ensure step_size doesn't exceed window size

    patches = []  # List to store patches for batch processing
    coords = []  # Store the coordinates of each patch
    num_patches = 0

    # Sliding window loop
    for y in range(0, wsi_image.shape[0] - window_size + 1, step_size):
        for x in range(0, wsi_image.shape[1] - window_size + 1, step_size):
            print(f"Collecting patch at coordinates ({x}, {y})", end='', flush=True)
            
            # Extract the current window (patch)
            patch = wsi_image[y:y+window_size, x:x+window_size, :]
            patches.append(patch / 255.0)  # Preprocess and normalize the patch
            coords.append((x, y))  # Store the coordinates
            num_patches += 1

            # When batch is full, process the batch
            if num_patches == batch_size:
                patches_input = np.array(patches)  # Convert list of patches to numpy array

                # Batch prediction
                predictions = model.predict(patches_input, verbose=None)

                # Update heatmap with predictions
                for i, (x_coord, y_coord) in enumerate(coords):
                    prediction = predictions[i][1]  # IDC positive class probability
                    heatmap[y_coord:y_coord+window_size, x_coord:x_coord+window_size] += prediction
                    prediction_count[y_coord:y_coord+window_size, x_coord:x_coord+window_size] += 1

                # Clear batch
                patches = []
                coords = []
                num_patches = 0

    # Process any remaining patches in the last batch
    if patches:
        patches_input = np.array(patches)
        predictions = model.predict(patches_input, verbose=None)

        for i, (x_coord, y_coord) in enumerate(coords):
            prediction = predictions[i][1]
            heatmap[y_coord:y_coord+window_size, x_coord:x_coord+window_size] += prediction
            prediction_count[y_coord:y_coord+window_size, x_coord:x_coord+window_size] += 1

    # Normalize the heatmap by dividing by the number of predictions at each pixel
    heatmap = np.divide(heatmap, prediction_count, out=np.zeros_like(heatmap), where=prediction_count!=0)

    # Resize the heatmap to the original image size
    sliding_heatmap = cv2.resize(heatmap, (wsi_image.shape[1], wsi_image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Normalize heatmap values between 0 and 1
    sliding_heatmap = (sliding_heatmap - np.min(sliding_heatmap)) / (np.max(sliding_heatmap) - np.min(sliding_heatmap))

    # Apply a color map to the heatmap for better visualization
    sliding_heatmap_color = cv2.applyColorMap(((np.ones_like(sliding_heatmap) - sliding_heatmap) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Blend the heatmap with the original WSI for overlay
    sliding_heatmap_overlay = cv2.addWeighted(wsi_image, 0.6, sliding_heatmap_color, 0.4, 0)

    important_sliding_heatmap_overlay = overlay_threshold_heatmap(wsi_image, sliding_heatmap, alpha=0.3)

    return sliding_heatmap_color, sliding_heatmap_overlay, important_sliding_heatmap_overlay

def save_full_patient_slides(ids = range(8863, 16896+1)):
    print("Stitching and Saving Slides...")
    dir = STITCHED_SLIDES_DIR
    for patient_id in ids:
        try:
            full_slide = stitch_slide_patches(str(patient_id))
            save_slide_image(dir, full_slide, f'{patient_id}_Full_Slide.png')
        except:
            pass
    print()
    return

def save_patched_heatmaps(model, ids = range(8863, 16896+1)):
    print("Stitching, Generating Patched Heatmaps, and Saving for Slides...")
    dir = PATCHED_HEATMAP_STITCHED_SLIDES_DIR
    for patient_id in ids:
        try:
            full_slide, raw_heatmap, smooth_heatmap, raw_overlayed_image, overlayed_image_smooth = stitch_and_apply_patched_heatmap(patient_id, model)

            save_slide_image(dir, overlayed_image_smooth, f'{patient_id}_Smooth_Patched_Heatmap_Overlay.png')
            # save_slide_image(dir, smooth_heatmap, f'{patient_id}_Smooth_Patched_Heatmap.png')

            save_slide_image(dir, raw_overlayed_image, f'{patient_id}_Blocked_Patched_Heatmap_Overlay.png')
            save_slide_image(dir, raw_heatmap, f'{patient_id}_Blocked_Patched_Heatmap.png')

            print(f"Stitched and Heatmapped Slide of Patient ID {patient_id}", end='\r')
        except FileNotFoundError:
            pass
    print()
    return

def save_sliding_heatmaps(model, ids = range(8863, 16896+1), step_size = SLIDING_WINDOW_STEP_SIZE):
    print("Stitching, Generating Sliding Heatmaps, and Saving for Slides...")
    dir = SLIDING_HEATMAP_STITCHED_SLIDES_DIR
    for patient_id in ids:
        try:
            heatmap, heatmap_overlay, important_heatmap_overlay = stitch_and_apply_sliding_heatmap(patient_id, model, step_size = step_size)

            save_slide_image(dir, heatmap_overlay, f'{patient_id}_Sliding_Heatmap_Overlay.png')
            save_slide_image(dir, heatmap, f'{patient_id}_Sliding_Heatmap.png')
            save_slide_image(dir, important_heatmap_overlay, f'{patient_id}_Sliding_Important_Heatmap_Overlay.png')

            print(f"Stitched and Heatmapped Slide of Patient ID {patient_id}", end='\r')
        except FileNotFoundError:
            pass
    print()


def stitch_and_apply_ground_truth_heatmap(patient_id, heatmap_weighting = HEATMAP_WEIGHTING):
    """
    Stitches slide and applies a heatmap to the stitched slide for a given patient using the model's predictions.
    Apples model prediction on each patch in the image
    """
    patient_folder = os.path.join(DATA_IMAGE_DIR, str(patient_id))

    # Initialize lists to store patches and their coordinates
    patches = []
    x_coords = []
    y_coords = []
    predictions = []
    
    # Iterate through both IDC negative (0) and IDC positive (1) folders
    for class_label in ['0', '1']:
        class_folder = os.path.join(patient_folder, class_label)
        for filename in os.listdir(class_folder):
            if filename.endswith(".png"):
                print("   "*50, end='\r')
                print("Patching Filename " + filename + " ...", end='\r')
                # Parse the filename to extract coordinates
                parts = filename.split('_')
                x = int(parts[2][1:])  # Extract X coordinate (e.g., x1351 -> 1351)
                y = int(parts[3][1:])  # Extract Y coordinate (e.g., y1101 -> 1101)
                
                # Load the image patch and preprocess for the model
                img = Image.open(os.path.join(class_folder, filename))
                img = np.array(img)
                
                # Pad the image if it's smaller than 50x50
                if img.shape != (PATCH_SIZE, PATCH_SIZE, 3):
                    img = np.pad(img, 
                                 ((0, PATCH_SIZE - img.shape[0]), (0, PATCH_SIZE - img.shape[1]), (0, 0)), 
                                 mode='constant', constant_values=0)
                
                # Store the prediction, image patch, and coordinates
                patches.append(img)
                x_coords.append(x)
                y_coords.append(y)

                # predictions.append(prediction[0][1])  # Confidence for IDC positive (class 1)
                predictions.append(int(class_label))
    
    # Calculate the size of the full slide
    max_x = max(x_coords) + PATCH_SIZE  # Add 50 to get the rightmost boundary
    max_y = max(y_coords) + PATCH_SIZE  # Add 50 to get the bottom boundary
    
    # Initialize an empty canvas for the full slide
    full_slide = np.ones((max_y, max_x, 3), dtype=np.uint8) * 255
    heatmap_overlay = np.zeros((max_y, max_x), dtype=np.float32)

    # Place each patch on the canvas and add prediction values to the heatmap
    for patch, x, y, pred in zip(patches, x_coords, y_coords, predictions):
        full_slide[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :] = patch
        heatmap_overlay[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = pred

    # Normalize the heatmap overlay to range [0, 1]
    min_val = np.min(heatmap_overlay)
    max_val = np.max(heatmap_overlay)
    range_val = max_val - min_val
    if range_val == 0:
        heatmap_overlay = np.zeros_like(heatmap_overlay)
    else:
        heatmap_overlay = (heatmap_overlay - min_val) / range_val


    # Apply a colormap to the heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap_overlay), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper overlay

    # Overlay the heatmap on the original slide
    overlayed_image = cv2.addWeighted(full_slide, 1-heatmap_weighting, heatmap, heatmap_weighting, 0)
    
    return full_slide, heatmap, overlayed_image


def save_ground_truth_heatmaps(ids = [8863]):
    print("Stitching, Generating Patched Heatmaps, and Saving for Slides...")
    dir = PATCHED_HEATMAP_STITCHED_SLIDES_DIR
    for patient_id in ids:
        try:
            full_slide, heatmap, overlayed_image = stitch_and_apply_ground_truth_heatmap(patient_id)

            save_slide_image(dir, overlayed_image, f'{patient_id}_Ground_Truth_Overlay.png')
            save_slide_image(dir, heatmap, f'{patient_id}_Ground_Truth_Heatmap.png')

            print(f"Stitched and Heatmapped Slide of Patient ID {patient_id}", end='\r')
        except FileNotFoundError:
            pass
    print()
    return