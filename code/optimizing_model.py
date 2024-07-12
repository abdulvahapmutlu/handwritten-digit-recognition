import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import numpy as np
import pickle

# Load preprocessed data
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

# Optimal hyperparameters from the search
optimal_dense_units = 64
optimal_learning_rate = 0.001

# Define the optimized CNN model
def build_optimized_model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(optimal_dense_units, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=optimal_learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

optimized_model = build_optimized_model()
optimized_model.summary()

# Create an ImageDataGenerator for data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(x_train.reshape(-1, 28, 28, 1))

# Train the optimized model
history = optimized_model.fit(datagen.flow(x_train.reshape(-1, 28, 28, 1), y_train, batch_size=32), 
                              epochs=20, 
                              validation_data=(x_test.reshape(-1, 28, 28, 1), y_test))

# Save the optimized model
model_save_path = 'optimized_mnist_model.h5'
optimized_model.save(model_save_path)
print(f'Model saved to {model_save_path}')

# Save the history object for evaluation
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
