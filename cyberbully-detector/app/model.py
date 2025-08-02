from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

def load_model(model_path='unitary/toxic-bert'):  # <--- public model
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model_path = "./fine_tuned_model"
    model.eval()
    return model, tokenizer

def predict_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()
    
    label_map = {0: "Not Cyberbullying", 1: "Cyberbullying"}
    return label_map[predicted_class], round(confidence, 4)
