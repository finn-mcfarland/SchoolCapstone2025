# SchoolCapstone2025
The 2025 Capstone project

# Cyberbullying Detection Web App

Use python version 3.9.6 and pip :)
assumes your PATH is set up correctly

## Overview

This project provides a simple web application for detecting cyberbullying in user-submitted comments. It combines a lightweight sequence classification model with a FastAPI backend and an HTML/JavaScript frontend that simulates a comment section. The system flags potentially harmful or bullying content based on model predictions.

## Features

- Text classification model trained to detect cyberbullying vs non-bullying comments  
- FastAPI backend serving a REST API (`/classify`) that runs the model on new inputs  
- Frontend (HTML/JS) that provides a mock comment section where users can type comments and receive immediate feedback  
- Configurable detection threshold so you can control sensitivity  

---

## The Files

- `Transformer.py`:  
    Training and data cleaning - current model is trained at 10 epochs
    
- `website_test.py`:  
    launches a gradio app with a minimal ui for this detector
        
- `testpredict.py`:  
    includes a variable that is a list of inputs to test, no website, runs in IDE

- `app.py` and `index.html`:
    the final website setup, guide for use below

--- 

## How to Use

### Installation

Clone this repository and install dependencies in a Python environment (Python 3.8+ recommended):
```bash
    git clone https://github.com/finn-mcfarland/SchoolCapstone2025.git
    cd SchoolCapstone2025
```

1. Create and activate a Python virtual environment (recommended):
    MACOS/Linux:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    Windows:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3. Install dependencies:
    there are probably other versions that work, but i Know these ones do on python 3.9.6
   you might have to update torch depending on your system?
    ```bash
    pip install --upgrade pip setuptools wheel
    pip install fastapi==0.115.0 uvicorn==0.30.6
    pip install transformers==4.44.2 torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
    pip install python-multipart==0.0.9
    pip install safetensors==0.4.5
    ```
    If you encounter OpenSSL warnings:
   ```bash
    pip install urllib3==1.26.18
    export SSL_CERT_FILE=$(python -m certifi)
   ```

5. Running the Server
```bash
    python -m uvicorn app:app --reload
```
The server will run on:

http://127.0.0.1:8000

4. Frontend

    Open index.html in your browser. It contains a simple mock comment section where users can type a message. Submissions are sent to the FastAPI backend, which returns the classification result.
        
    If the comment is classified as bullying (above threshold), it will be highlighted as such
    
    How It Works

    Model
    A sequence classification model (fine-tuned transformer) is loaded from Hugging Face Transformers.
    It takes text as input and produces logits for two classes:

        Class 0 = Not Bullying

        Class 1 = Bullying

    Classification

        The backend applies a softmax to convert logits into probabilities

        By default, the higher probability class is returned

        An optional detection threshold can be applied to require higher confidence before labeling as bullying

Dependencies
FastAPI – web framework for serving the API
Uvicorn – ASGI server to run FastAPI
Transformers – pretrained NLP models
Torch - deep learning backend for Transformers

References
https://ieeexplore.ieee.org/document/9908898 - access 30/jul/2025

accessed 31/jul/2025:
https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset
https://data.mendeley.com/datasets/wmx9jj2htd/2
https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification?resource=download
