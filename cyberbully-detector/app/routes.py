from flask import Blueprint, request, jsonify
from .model import load_model, predict_text

main = Blueprint('main', __name__)

model, tokenizer = load_model()

@main.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    prediction, confidence = predict_text(text, model, tokenizer)
    return jsonify({'text': text, 'prediction': prediction, 'confidence': confidence})
