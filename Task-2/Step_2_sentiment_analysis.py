#**********************************************************************************************************
# This is step-2 file
# This code trains the sentiment analysis csv dataset to build the model
# PLease provide the input path inorder to run/test the code.
# Although, the code is pre trained and tested and model is stored as sentiment_model.pkl, its attached with the file.
#***********************************************************************************************************


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the sentiment analysis dataset
#data = pd.read_csv(r'C:\Users\srikr\OneDrive\Desktop\Epoch Tasks\TASK-2\sentiment_analysis_dataset.csv')
data = pd.read_csv(input("please enter the path to file: "))

# Preview the data to check the column names and content
print(data.head())
print(data.columns)

# Ensure the correct columns are selected for text and labels
X = data['line']
y = data['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with a CountVectorizer and a Multinomial Naive Bayes classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Perform cross-validation to evaluate the model
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f'Cross-validation accuracy scores: {cv_scores}')
print(f'Mean cross-validation accuracy: {cv_scores.mean()}')

# Train the model on the entire training set
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy}')
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'sentiment_model.pkl')
