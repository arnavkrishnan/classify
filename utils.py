# utils.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the model and tokenizer once
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Function to make predictions
def predict(text):
    """
    Given input text, predict the class label using the pre-trained BERT model.
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the prediction (logits) and convert to probabilities
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return prediction
