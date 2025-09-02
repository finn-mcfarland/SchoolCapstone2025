import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load trained model + tokenizer
model_path = "./cyberbully_detector"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Ensure model is in eval mode
model.eval()

# Example test sentences
examples = [
    "KILL YOURSELF",
    "LOVE YOURSELF",
    "NIGGER"
]

# Tokenize
inputs = tokenizer(
    examples,
    padding=True,
    truncation=True,
    max_length=220,
    return_tensors="pt"
)

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

# Decode predictions
id2label = model.config.id2label
for text, pred_id in zip(examples, predictions):
    print(f"Text: {text}")
    print(f"Prediction: {id2label[int(pred_id)]}")
    print("-" * 40)
