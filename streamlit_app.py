# streamlit_app.py
import streamlit as st
from utils import predict  # Import the prediction function from utils.py

# Streamlit UI elements
st.title("BERT NLP Model for Text Classification")
st.write("Enter some text to classify:")

# Text input from user
user_input = st.text_area("Text", "")

# Button to make prediction
if st.button("Classify"):
    if user_input:
        prediction = predict(user_input)  # Call the predict function from utils.py
        st.write(f"Predicted Class: {prediction}")
    else:
        st.write("Please enter some text to classify.")
