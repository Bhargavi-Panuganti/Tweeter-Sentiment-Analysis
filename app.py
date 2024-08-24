import streamlit as st
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import nltk

# Download the NLTK stopwords
nltk.download('stopwords')

# Load the saved model and vectorizer
model_filename = 'trained_model.sav'
vectorizer_filename = 'vectorizer.sav'

with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_filename, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize the Porter Stemmer
port_stem = PorterStemmer()

def preprocess_text(text):
    """Function to preprocess the input text for sentiment analysis"""
    # Remove non-alphabetic characters
    stemmed_content = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    stemmed_content = stemmed_content.lower()
    # Split into words
    stemmed_content = stemmed_content.split()
    # Stemming and removing stopwords
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    # Rejoin words
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def predict_sentiment(text):
    """Function to predict the sentiment of a given text"""
    # Preprocess the input text
    processed_text = preprocess_text(text)
    # Transform the text using the loaded vectorizer
    transformed_text = vectorizer.transform([processed_text])
    # Predict sentiment using the loaded model
    prediction = model.predict(transformed_text)
    return "Negitive" if prediction[0] == 0 else "Positive"

# Streamlit app interface
st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet below and find out whether it's positive or negative!")

# User input
tweet = st.text_area("Enter Tweet:")

if st.button("Analyze"):
    if tweet:
        # Predict sentiment
        sentiment = predict_sentiment(tweet)
        # Display the result
        st.write(f"The sentiment of the tweet is: **{sentiment}**")
    else:
        st.write("Please enter a tweet.")

