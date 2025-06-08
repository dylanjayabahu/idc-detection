from data import *
from homogenous_oversampling import *
from image_processing import *
from model import * 
from parameters import *

def main():
    ##preprocess data
    save_full_patient_slides()
    save_homogenous_oversamples()
    save_pngs_to_np_dataset()

    ##load data
    train_ds, validation_ds, x_test, y_test = load_data(
        batch_size=BATCH_SIZE,
        show_data_stats=True,
        augment_data=True,
        num_aug_visualizations=0)

    ##load model
    if MODEL_NAME:
        model_file = MODEL_NAME if '.' in MODEL_NAME else MODEL_NAME + ".keras"
        model = load_model(model_file, optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, decay=LEARNING_RATE_DECAY))
    else:
        model = make_model(show_summary=True, model_number=MODEL_NUMBER)

    ##train model
    if SHOULD_TRAIN:
        model, model_filename = train_model(model, train_ds, validation_ds, 
                                            epochs=NUM_EPOCHS, early_stopper_patience=EARLY_STOPPER_PATIENCE)
        save_model(model, name=model_filename + ".keras")


    ##test model
    test_model(x_test, y_test, model)

    ##use model
    save_patched_heatmaps(model)
    save_sliding_heatmaps(model, step_size = SLIDING_WINDOW_STEP_SIZE) #lower step sizes are more computationally intense

if __name__ == "__main__":    
    main()