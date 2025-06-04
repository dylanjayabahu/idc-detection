##SETUP PARAMETERS
MODEL_NUMBER = 8 #which model architecture to use (see model_versions.py)
SHOULD_TRAIN = True #if main should execute model training
MODEL_NAME = None #name of already to saved model, if applicable

##DATA PARAMETERS
PATCH_SIZE = 50 #size of image patches
DEFAULT_SEED = 42 #set a seed for reproducability
NUM_TRAIN = 84 #num patient ids in the train set
NUM_VALIDATION = 29 #num patient ids in the train set
NUM_TEST = 49 #num patient ids in the train set


##TRAINING PARAMETERS
LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY = 1e-6
BATCH_SIZE = 32 #batch size for model training
NUM_EPOCHS = 50
EARLY_STOPPER_PATIENCE = 10 #num epochs to continue after lowest loss has been achievved (-1 = no early stopping)
OPTIMIZER = 'adam'

SLIDING_WINDOW_STEP_SIZE = 5 #step size for the sliding window heatmap generation. Lower step is more detailed but more
OVERLAY_THRESHOLD = 0.3 #alpha-value for the threshold heatmap overlay 
HEATMAP_WEIGHTING = 0.4 #transparency weighting for the transparent heatmap overlay
SLIDING_WINDOW_BATCH_SIZE = 256 #batch size for sliding window predictions

##additional less important parameters are found as default parameters for functions