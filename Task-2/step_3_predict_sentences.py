#*********************************************************************************************************
# This is Step-3 file.
# This File Predicts the sentences from the input image.
# It also predicts the sentiment associated with the predicted text.
# PLease provide the path to the images in order to test this file.
# Code is implemented in such a way that image is broken into pieces 28x28 to predict each character and combine to form the sentence.
#**********************************************************************************************************



import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import joblib
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the label encoderr
with open('label_encoder.npy', 'rb') as f:
    classes = np.load(f, allow_pickle=True)

# Load the trained model
mnist_model = load_model('mnist_model.h5')

# Load the trained sentiment analysis model
sentiment_model = joblib.load('sentiment_model.pkl')

def preprocess_image(image):
    # Normalize and reshapje the image to fit the model input
    image = image.astype(np.float32) / 255.0  
    image = np.expand_dims(image, axis=0)  
    image = np.expand_dims(image, axis=-1)  
    return image

def predict_character(image, model, classes):
    #predict the character in the image
    image = preprocess_image(image)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_character = classes[predicted_class][0]
    return predicted_character

def segment_and_predict(image_path, model, classes):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    
    # Define patch size
    patch_size = 28
    
    # Get image dimensions
    height, width = image.shape
    
    predicted_sentence = ""
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            # Extract the patch
            patch = image[y:y+patch_size, x:x+patch_size]
            
            # Ensure the patch is 28x28
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                # Predict the character in the patch
                character = predict_character(patch, model, classes)
                if character == 'P':  
                    character = ' '
                predicted_sentence += character
    
    return predicted_sentence

def predict_sentiment(sentence):
    return sentiment_model.predict([sentence])[0]

if __name__ == "__main__":
    # Define the path to the image
    #image_path =  r'C:\Users\srikr\OneDrive\Desktop\Epoch Tasks\TASK-2\target_images\line_3.png'
    image_path = input("PLease enter the path to image: ")

    # Predict the sentence
    predicted_sentence = segment_and_predict(image_path, mnist_model, classes)
    print(f'Predicted sentence: {predicted_sentence}')
   
   
    # Perform sentiment analysis on the predicted sentence
    predicted_sentiment = predict_sentiment(predicted_sentence)
    print(f'Predicted Sentiment: {predicted_sentiment}')