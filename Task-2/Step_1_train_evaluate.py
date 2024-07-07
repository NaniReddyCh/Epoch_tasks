#***********************************************************************************************************
# This is a Step 1 file
# This File Trains and Evaluates the alphabets_28x28.csv dataset
# It performs total of 10 epochs to complete the model building, training and testing.
# Altough measures where taken, It takes significant amount of time to complete the process (5-10 min) as the dataset is largeenough to load and test.
# Please download the file and provide the path to dataset as in your computer inorder to test the program.
# The model which is built from this program(titled: mnist_model.h5, lable encoder: label_encoder.npy) is attached along with the file if its not possible/taking lot of time to build the model.
#***********************************************************************************************************

# train_evaluate.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#csv_path = r'C:\Users\srikr\OneDrive\Desktop\Epoch Tasks\TASK-2\alphabets_28x28.csv'
csv_path = input("please provide the path to csv file: ")


def load_data(csv_path):
    # Load the data from your CSV file
    alpha28_csv = pd.read_csv(csv_path, low_memory=False)

    # Remove corrupted rows (rows with NaN values)
    alpha28_csv = alpha28_csv.dropna()

    # Separate features and labels
    X = alpha28_csv.drop('label', axis=1)
    y = alpha28_csv['label']

    # Convert all data to numeric
    X = X.apply(pd.to_numeric)

    # Reshape the data to fit the model (n_samples, 28, 28, 1)
    X = X.values.reshape(-1, 28, 28, 1)

    # Normalize the data
    X = X / 255.0

    # Encode the labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y, label_encoder

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_and_evaluate(csv_path, test_size=0.2, epochs=10, batch_size=32):
    X, y, label_encoder = load_data(csv_path)
    num_classes = len(label_encoder.classes_)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # Build and train the model
    mnist_model = create_model(input_shape=(28, 28, 1), num_classes=num_classes)
    mnist_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

    # Evaluate the model
    loss, accuracy = mnist_model.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy}')

    # Save the model and the label encoder
    mnist_model.save('mnist_model.h5')
    with open('label_encoder.npy', 'wb') as f:
        np.save(f, label_encoder.classes_)

    return mnist_model, label_encoder

# Define the CSV path and call the train_and_evaluate function directly
train_and_evaluate(csv_path)
