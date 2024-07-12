import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Load preprocessed data
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

# Load the model
model = tf.keras.models.load_model('optimized_mnist_model.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test.reshape(-1, 28, 28, 1), y_test)
print(f'\nTest accuracy: {test_acc}')

# Predictions
y_pred = model.predict(x_test.reshape(-1, 28, 28, 1))
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_classes))

# Confusion matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred_classes))

# Load the history object
with open('training_history.pkl', 'rb') as f:
    history = pickle.load(f)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')

plt.show()
