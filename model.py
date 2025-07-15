import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import tensorflow as tf

from data_config import TENSORBOARD_DIR, SAVED_MODELS_DIR, CONFUSION_MATRIX_DIR
from model_versions import *
from parameters import *
    
def make_model(show_summary=True, model_number = MODEL_NUMBER):
    """
    creates a new model with architecture specified by model_number
    """
    models = [make_model1, make_model2, make_model3, make_model4, make_model5, make_model6, make_model7, make_model8]
    return models[model_number-1](show_summary=show_summary)

def train_model(model, train_ds, validation_ds, epochs=1, early_stopper_patience = EARLY_STOPPER_PATIENCE, 
                model_filename = "New_model.keras", use_tensorboard = True, use_checkpoints = True):
    
    
    train_callbacks = []

    if early_stopper_patience > 0:
        ##early stopper stops training if validation loss stops decreasing
        early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', 
                                                        min_delta=0, 
                                                        patience=early_stopper_patience, 
                                                        verbose=1,
                                                        mode='max',
                                                        restore_best_weights=True,)
        
        train_callbacks.append(early_stopper)
    

    if use_tensorboard:
        path = os.path.join(TENSORBOARD_DIR, model_filename+"_tensorboard_data") 

        assert not os.path.exists(path)

        os.makedirs(path) 
        log_dir = TENSORBOARD_DIR+ model_filename+"_tensorboard_data"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                            histogram_freq=1, 
                                                            write_graph=True,
                                                            # update_freq='batch'
                                                            )
        train_callbacks.append(tensorboard_callback)

    if use_checkpoints:
        checkpoint_dir = os.path.join(SAVED_MODELS_DIR, model_filename + '_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, model_filename + '_checkpoint_{epoch:04d}.keras'),
            save_freq='epoch',
        )

        train_callbacks.append(checkpoint_callback)

    #fit model
    model.fit(train_ds,
            epochs=epochs, 
            verbose = 1,
            validation_data=validation_ds,
            callbacks=train_callbacks) 

    return model, model_filename

def save_model(model, name = ""):
    if name != "":    
        model.save(os.path.join(SAVED_MODELS_DIR, name))
        print("Model saved under filename:", name)
    else:
        print("No filename entered. Not saving model.")

def save_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    """
    Generate and save a confusion matrix plot with a color bar.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot the confusion matrix using seaborn heatmap
    plt.figure(figsize=(6, 4))
    heatmap = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                           xticklabels=['IDC(-)', 'IDC(+)'], yticklabels=['IDC(-)', 'IDC(+)'])
    
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    
    # Add color bar
    plt.colorbar(heatmap.collections[0])  # Use the heatmap's collections for color bar

    # Save the plot as an image
    plt.savefig(os.path.join(CONFUSION_MATRIX_DIR, filename))
    plt.close()

def test_model(x_data, y_data, model):
    """
    Returns tuple containing accuracy, balanced accuracy, precision, recall, specificity, and F1-score.
    """
    
    # Get model predictions
    y_pred_prob = model.predict(x_data)

    # If y_data is one-hot encoded, convert to binary labels
    if len(y_data.shape) > 1 and y_data.shape[1] > 1:
        y_data = np.argmax(y_data, axis=1)  # Convert one-hot to binary labels

    # Convert predictions to binary labels based on a threshold (0.5 by default)
    y_pred = np.argmax(y_pred_prob, axis=1)  # Convert predicted probabilities to binary labels

    # Calculate confusion matrix components: tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_data, y_pred).ravel()

    # Calculate metrics
    accuracy = accuracy_score(y_data, y_pred)
    balanced_acc = balanced_accuracy_score(y_data, y_pred)
    precision = precision_score(y_data, y_pred)
    recall = recall_score(y_data, y_pred)  # Also known as True Positive Rate
    specificity = tn / (tn + fp)  # True Negative Rate
    f1 = f1_score(y_data, y_pred)

    # Print the metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (True Positive Rate): {recall:.4f}")
    print(f"Specificity (True Negative Rate): {specificity:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Save the confusion matrix
    save_confusion_matrix(y_data, y_pred, filename="confusion_matrix.png")

    return accuracy, balanced_acc, precision, recall, specificity, f1

def load_model(load_model_name, show_summary = True, optimizer = OPTIMIZER):
    try:
        model = tf.keras.models.load_model(os.path.join(SAVED_MODELS_DIR, load_model_name))
        model.compile(optimizer=optimizer,
                loss='categorical_crossentropy', 
                metrics=['accuracy']) 
        if show_summary:
            model.summary()
        return model
    

    except:
        ##if user has inputted an invalid filename, ask them to try again
        print("Invalid filename. Enter a valid model saved in " + SAVED_MODELS_DIR)
        command = input("Enter a valid name for a saved model: ")
        if '.' in command:
            return load_model(command)
        else:
            return load_model(command + ".keras")