from dotenv import load_dotenv
load_dotenv()

import base64
import logging
import os
from io import BytesIO

from flask import Flask, jsonify, request
from PIL import Image

from device import get_device, use_gpu_enabled
from inference import load_inference_model, predict_image

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

USE_GPU = os.getenv('USE_GPU', 'false').lower() in ('true', '1', 't')
CHECKPOINT_PATH = os.getenv('CHECKPOINT', 'datasets/oxford102/model/resnet152_oxford102.pth')

device = get_device(use_gpu_enabled(USE_GPU))

try:
    model, CLASS_NAMES, transform, MODEL_META = load_inference_model(
        CHECKPOINT_PATH, device=device, use_gpu=use_gpu_enabled(USE_GPU),
    )
    NUM_CLASSES = len(CLASS_NAMES)
    logging.info('Loaded model from %s (%d classes)', CHECKPOINT_PATH, NUM_CLASSES)
except FileNotFoundError:
    logging.warning('Checkpoint not found at %s — endpoints will fail until trained', CHECKPOINT_PATH)
    model = None
    CLASS_NAMES = []
    transform = None
    MODEL_META = {}
    NUM_CLASSES = 0


@app.route('/predict', methods=['POST'])
def predict():
    """Predict flower class for a base64-encoded image (JSON field `image`)."""
    if model is None:
        return jsonify({'error': 'Model checkpoint not loaded'}), 503

    try:
        data = request.get_json(force=True)
        image_bytes = base64.b64decode(data['image'])
        image = Image.open(BytesIO(image_bytes)).convert('RGB')

        predictions = predict_image(
            model, transform, CLASS_NAMES, image, device, top_k=5,
        )

        response = {
            'prediction': predictions[0],
            'top_5_predictions': predictions,
            'confidence': predictions[0]['confidence'],
        }
        return jsonify(response)

    except Exception as e:
        logging.error('Error during prediction: %s', e)
        return jsonify({'error': 'Error during prediction'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'num_classes': NUM_CLASSES,
    }), 200


@app.route('/classes', methods=['GET'])
def list_classes():
    return jsonify({
        'total_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'meta': MODEL_META,
    }), 200


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
