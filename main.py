# import Libraries and load the model
import numpy as np
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model



# Load the IMDB dataset and get word index
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}
# Load the pre-trained model with ReLU activation 
model=load_model('simple_rnn_imdb.h5')


#to help with this we create 2 helper functions

# 1st function to decode_reviews which we saw earliar
def decode_review(encode_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encode_review])

#Function to preprocess user input
def preprocess_text(text):
    words=text.lower().split()
    #encode the review given by user in index code
    encode_review=[word_index.get(word,2)+3 for word in words]
    #then pad it then only can give it to our model
    padded_review=sequence.pad_sequences([encode_review],maxlen=500)
    return padded_review



## streamlit app
import streamlit as st

st.title ('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# user input
user_input=st.text_area('Movie Review')

if st.button('Classify'): # when we press the button it should do the classification
    #by doing all below steps
    preprocess_input=preprocess_text(user_input)
    #make prediction
    prediction=model.predict(preprocess_input)

    sentiments='Positive' if prediction[0][0]>0.5 else 'Negative'

    #Display result
    st.write(f'Sentiment:{sentiments}')
    st.write(f'Prediction Score:{prediction[0][0]}')

else :
    st.write('Please write movie review')