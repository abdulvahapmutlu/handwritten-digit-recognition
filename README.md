# Handwritten Digit Recognition using CNN and MNIST Dataset

This repository contains a project focused on recognizing handwritten digits using the MNIST dataset. The project involves building, training, and optimizing a Convolutional Neural Network (CNN) to achieve high accuracy in digit recognition. The project involves data preprocessing, model training, hyperparameter tuning, and evaluation of the model's performance.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)

## Introduction
Handwritten Digit Recognition is a classic problem in the field of computer vision and machine learning. This project leverages deep learning techniques, specifically Convolutional Neural Networks (CNNs), to accurately classify digits from 0 to 9 based on images from the MNIST dataset.

## Dataset
The MNIST dataset contains 70,000 grayscale images of handwritten digits, each 28x28 pixels in size. The dataset is divided into 60,000 training images and 10,000 test images. Each image is labeled with the corresponding digit (0-9).

## Installation
To run this project, you need to have Python installed along with the necessary libraries. You can install the required libraries using the following command:
```
pip install -r requirements.txt
```
Or you can manually install the dependencies below:
- TensorFlow
- NumPy
- Matplotlib
- Keras Tuner
- scikit-learn

## Usage
1. **Data Preprocessing**: Run `data_preprocessing.py` to load and preprocess the MNIST dataset.
2. **Model Training**: Run `model_training.py` to build, train, and save the CNN model.
3. **Hyperparameter Tuning**: Run `hyperparameter_tuning.py` to perform hyperparameter tuning and find the optimal parameters.
4. **Evaluation**: Run `evaluation_metrics.py` to evaluate the trained model and generate metrics.

## Results
The final optimized model achieved a test accuracy of approximately 99.4%.
The model achieved a high accuracy on the test set. Demonstrating the effectiveness of the CNN model. Detailed evaluation metrics and plots are provided to analyze the model's performance.
Additionally, training and validation accuracy/loss plots are generated to visualize the model's performance over epochs.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Conclusion
This project demonstrates the effectiveness of Convolutional Neural Networks (CNNs) in recognizing handwritten digits. Through data preprocessing, model training, hyperparameter tuning, and evaluation, a high accuracy model was achieved. Future work could involve experimenting with different architectures, data augmentation techniques, and other datasets to further improve performance.

## Acknowledgements
- The MNIST dataset is publicly available and can be accessed with 'tf.keras.datasets.mnist'.
- This project utilizes TensorFlow and Keras for building and training the CNN model.

