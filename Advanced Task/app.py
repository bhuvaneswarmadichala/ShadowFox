from flask import Flask, request, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

app = Flask(__name__)

# Load the tokenizer and model
model_dir = "bert_sentiment_model"
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Home route with form
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form["text"]
        
        # Tokenize input
        encoding = tokenizer(text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
        
        # Map prediction to sentiment
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        result = sentiment_map[prediction]
        
        return render_template("index.html", result=result)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)