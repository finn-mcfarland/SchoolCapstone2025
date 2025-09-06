import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch

model = AutoModelForSequenceClassification.from_pretrained("./cyberbully_detector")
tokenizer = AutoTokenizer.from_pretrained("./cyberbully_detector")

labels = ["Bullying", "Not Bullying"] 

def classify(text):
    inputs = tokenizer([text], return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()
    return labels[prediction], {labels[i]: float(probs[0][i]) for i in range(len(labels))}

iface = gr.Interface(fn=classify, inputs="text", outputs=["label", "json"])
iface.launch(share=True)
