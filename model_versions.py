import tensorflow as tf

from parameters import PATCH_SIZE
from metrics import *

def make_model1(show_summary=True):
    ##make model

    model=tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(6,(3,3),input_shape=(PATCH_SIZE,PATCH_SIZE,3),activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(10,(3,3),activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

    model.add(tf.keras.layers.Flatten()) 

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(80, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy']) 
    
    ##show summary if requested
    if show_summary:
        model.summary()

    return model

def make_model2(show_summary=True):
    model = tf.keras.models.Sequential()

    # Add convolutional layers
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(PATCH_SIZE, PATCH_SIZE, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Global Average Pooling
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Dense layers
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    if show_summary:
        model.summary()

    return model

def make_model3(show_summary=True):
    model = tf.keras.models.Sequential()

    # Add convolutional layers
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(PATCH_SIZE, PATCH_SIZE, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Global Average Pooling
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Dense layers
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    if show_summary:
        model.summary()

    return model

def make_model4(show_summary=True):
    model = tf.keras.models.Sequential()

    # Add convolutional layers
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(PATCH_SIZE, PATCH_SIZE, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Global Average Pooling
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Dense layers
    model.add(tf.keras.layers.Dense(128, activation=None))
    model.add(tf.keras.layers.PReLU()) ##try prely activation instead of relu
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation=None))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    if show_summary:
        model.summary()

    return model

def make_model5(show_summary=True):
    model = tf.keras.models.Sequential()

    # Add convolutional layers
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(PATCH_SIZE, PATCH_SIZE, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Global Average Pooling
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Dense layers
    model.add(tf.keras.layers.Dense(128, activation=None))
    model.add(tf.keras.layers.PReLU()) ##try prelu activation instead of relu
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation=None))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

    # Use a lower initial learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # Default is 0.001

    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    if show_summary:
        model.summary()

    return model

def make_model6(show_summary=True):
    model = tf.keras.models.Sequential()

    # Add convolutional layers
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(PATCH_SIZE, PATCH_SIZE, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Global Average Pooling
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Dense layers
    model.add(tf.keras.layers.Dense(128, activation=None))
    model.add(tf.keras.layers.PReLU()) ##try prelu activation instead of relu
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation=None))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

    # Use a lower initial learning rate
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=100000,  # Number of steps after which learning rate decays
        decay_rate=0.96,     # The factor by which the learning rate will decay
        staircase=True)      # Whether to apply decay in a discrete staircase function

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    if show_summary:
        model.summary()

    return model

def make_model7(show_summary=True):
    model = tf.keras.models.Sequential()

    # Add convolutional layers
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(PATCH_SIZE, PATCH_SIZE, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Global Average Pooling
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Dense layers
    model.add(tf.keras.layers.Dense(128, activation=None))
    model.add(tf.keras.layers.PReLU()) ##try prelu activation instead of relu
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation=None))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[6],  # After 6 epochs
        values=[0.0001, 0.00005]  # Start with 0.0001, then 0.00005 after 6 epochs
    )
    # Use a lower initial learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    if show_summary:
        model.summary()

    return model

def make_model8(show_summary=True):
    model = tf.keras.models.Sequential()

    # Add convolutional layers with increased filters and different activation functions
    model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(PATCH_SIZE, PATCH_SIZE, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))

    # Global Average Pooling
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Dense layers with more units
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))  # Changed to softmax for multi-class classification

    # Use a lower initial learning rate with weight decay
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6)

    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', 'Precision', 'Recall', F1Score()])

    if show_summary:
        model.summary()

    return model