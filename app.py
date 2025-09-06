from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CommentRequest(BaseModel):
    comment: str

model = AutoModelForSequenceClassification.from_pretrained("./cyberbully_detector")
tokenizer = AutoTokenizer.from_pretrained("./cyberbully_detector")
labels = ["Bullying", "Not Bullying"]

@app.post("/classify")
async def classify(req: CommentRequest):
    inputs = tokenizer([req.comment], return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    prediction = probs.argmax(dim=-1).item()
    return {"comment": req.comment, "result": labels[prediction]}
