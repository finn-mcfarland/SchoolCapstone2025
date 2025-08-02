# SchoolCapstone2025
The 2025 Capstone project

https://ieeexplore.ieee.org/document/9908898 - access 30/jul/2025

accessed 31/jul/2025:
https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset
https://data.mendeley.com/datasets/wmx9jj2htd/2
https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification?resource=download



# Cyberbully Detector Flask App

## Overview

This project adds an **Autonomous Cyberbullying Detection API** built with Flask and a BERT-based model. The app exposes an endpoint for predicting whether a given text contains cyberbullying content.

Key features:

- Loads a pre-trained BERT model (`unitary/toxic-bert`) for text classification
- Provides a `/predict` POST endpoint accepting JSON input with text
- Returns prediction label (`Cyberbullying` or `Not Cyberbullying`) and confidence score
- Structured using Flask Blueprints for modularity
---

## What Was Added

- `model.py`:  
    Implements model loading and text prediction using HuggingFace transformers and PyTorch.
    
- `routes.py`:  
    Defines the Flask Blueprint with the `/predict` route for inference and a simple root route to confirm the API is running.
    
- Updated `__init__.py` to register the blueprint and create the Flask app.
    
- `run.py`:  
    Entry point to launch the Flask development server.
    

---

## How to Use

### Setup

1. Create and activate a Python virtual environment (recommended):
    
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    
2. Install dependencies:
    
    ```bash
    pip install -r requirements.txt
    ```
    
    > Make sure `transformers`, `torch`, `flask` are included in `requirements.txt`. (should be but check)

---

### Running the App

make sure to be in the folder of the 'run.py' file 
```bash
python run.py
```

The app will start on:  
`http://127.0.0.1:5000`

Visit this URL in a browser to see the root message confirming the server is running.

---

### Testing the Prediction Endpoint

Send a POST request to `/predict` with JSON body containing a `text` field. For example:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "You are an idiot"}'
```

Expected JSON response:

```json
{
  "text": "You are an idiot",
  "prediction": "Not Cyberbullying",
  "confidence": 0.91
}
``` 

---

## Notes

- The model uses a publicly available BERT checkpoint (`unitary/toxic-bert`) which can be replaced with a fine-tuned version later.
- the outcome should always be 'not cyberbulling' because we are not using a fine tuned model
- The app runs in Flask’s development mode; for production deployment, a WSGI server is recommended.
- Make sure to test the `/predict` endpoint with various texts to understand model behavior. after we implement an actual fine tuned model
