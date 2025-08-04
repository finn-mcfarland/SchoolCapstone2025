import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch.nn as nn
import torch

# Label mappings
label2id = {
    "not_cyberbullying": 0,
    "other_cyberbullying": 1
}
id2label = {v: k for k, v in label2id.items()}

# Load dataset and clean column names
df = pd.read_csv("Collated.csv", encoding="latin1")
df.columns = df.columns.str.strip().str.lower()  # standardize column names

# Drop rows with missing text or label
df = df.dropna(subset=['text', 'label'])

# Map string labels to integers
df['label'] = df['label'].map(label2id)

# Drop rows with unmapped labels (NaN after mapping)
df = df.dropna(subset=['label'])

# Convert labels to int
df['label'] = df['label'].astype(int)

# Split dataset into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"]
)

# Create Hugging Face Datasets
train_dataset = Dataset.from_dict({"text": train_texts.tolist(), "labels": train_labels.tolist()})
val_dataset = Dataset.from_dict({"text": val_texts.tolist(), "labels": val_labels.tolist()})


# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("unitary/toxic-bert")

# Tokenize datasets
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.map(lambda x: {"labels": int(x["labels"])})
val_dataset = val_dataset.map(lambda x: {"labels": int(x["labels"])})

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels") 
        
        if labels is None:
            raise ValueError(f"No 'labels' found in inputs: {inputs.keys()}")

        if isinstance(labels, torch.Tensor):
            if labels.dim() > 1:
                labels = labels.squeeze(-1)
            labels = labels.long()

        outputs = model(**inputs) 
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss



# Load model with correct label mappings
model = BertForSequenceClassification.from_pretrained(
    "unitary/toxic-bert",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Initialize Trainer
trainer = MyTrainer( 
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print(train_dataset[0]["labels"], type(train_dataset[0]["labels"]))

# Train and save the model
trainer.train()

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
