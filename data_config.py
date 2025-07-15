import os

PROJECT_DIR = "/Users/dylan-jayabahu/Desktop/Code/Projects/Breast_Cancer_Detection_AI"

DATA_NP_DIR = os.path.join(PROJECT_DIR, "data/histology_image_full_dataset") 
DATA_IMAGE_DIR = os.path.join(PROJECT_DIR, "data/IDC_regular_ps50_idx5")
DATA_AUGMENTATIONS_DIR = os.path.join(PROJECT_DIR, "augmentations")
STITCHED_SLIDES_DIR = os.path.join(PROJECT_DIR, "data/stitched_slides")
PATCHED_HEATMAP_STITCHED_SLIDES_DIR = os.path.join(PROJECT_DIR, "data/patched_heatmap_slides")
SLIDING_HEATMAP_STITCHED_SLIDES_DIR = os.path.join(PROJECT_DIR, "data/sliding_heatmap_slides")
TENSORBOARD_DIR = os.path.join(PROJECT_DIR, "tensorboard_logs/")
SAVED_MODELS_DIR = os.path.join(PROJECT_DIR, "saved_models/")
CONFUSION_MATRIX_DIR = os.path.join(PROJECT_DIR, "confusion_matrices")