import re
import string
import numpy as np
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate


def clean_text(text):
    if text is None:
        return None
    text = re.sub(r"http\S+|www\S+", "", text)  # URLs
    text = re.sub(r"@\w+", "", text)            # mentions
    text = re.sub(r"#\w+", "", text)            # hashtags
    text = text.translate(str.maketrans("", "", string.punctuation))  # punctuation
    text = re.sub(r"\s+", " ", text).strip()    # extra whitespace
    return text

#load
dataset = load_dataset("csv", data_files={"train": "data.csv"}, column_names=["text", "label"])

# drop missing + clean
dataset = dataset.filter(lambda x: x["text"] is not None and x["label"] is not None)
dataset = dataset.map(lambda x: {"text": clean_text(x["text"])})
dataset = dataset.filter(lambda x: x["text"] != "")

# labels
labels_list = sorted(list(set(dataset["train"]["label"])))
label_feature = ClassLabel(names=labels_list)  
label_to_id = {label: i for i, label in enumerate(labels_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

def encode_labels(example):
    example["label"] = label_to_id[example["label"]]
    return example

dataset = dataset.map(encode_labels)
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

#tokenisation
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding=False,   # let data collator handle padding
        max_length=220   # slightly longer than 128 for fun
    )

dataset = dataset.map(preprocess, batched=True, remove_columns=["text"])

#the model itself
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels_list),
    id2label=id_to_label,
    label2id=label_to_id
)

#metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
    }

#training setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=64,
    num_train_epochs=10,               
    weight_decay=0.01,
    warmup_ratio=0.1,                
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#the trainerrrr
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

#actually do anything
trainer.train()

trainer.save_model("./cyberbully_detector")
tokenizer.save_pretrained("./cyberbully_detector")
