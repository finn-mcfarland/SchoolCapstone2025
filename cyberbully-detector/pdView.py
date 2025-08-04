import pandas as pd

# Load CSV and clean columns
df = pd.read_csv("Collated.csv", encoding="latin1")
df.columns = df.columns.str.strip().str.lower()  # 'text', 'label'

# Drop rows with missing text or label
df = df.dropna(subset=['text', 'label'])

# Map string labels to integers (adjust keys to match your dataset exactly)
label2id = {
    "not_cyberbullying": 0,
    "other_cyberbullying": 1  # adjust this key to your actual label names
}

df['label'] = df['label'].map(label2id)

# Drop rows with unmapped labels (NaN after mapping)
df = df.dropna(subset=['label'])

# Convert labels to int type
df['label'] = df['label'].astype(int)

print(df.head())
print(df['label'].value_counts())
