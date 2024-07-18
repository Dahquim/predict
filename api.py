from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
from utils import load_model
from torchvision.models import resnet152
import os
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

filename = os.getenv('MODEL_PATH', "resources\\resnet152_weights_best_acc.tar")  # pre-trained model path
num_classes = int(os.getenv('NUM_CLASSES', 1081))  # number of classes in the model
use_gpu = os.getenv('USE_GPU', False).lower() in ('true', '1', 't')  # load weights on the gpu

model = resnet152(num_classes=num_classes)
load_model(model, filename=filename, use_gpu=use_gpu)  # load the model
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        image_data = data['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output.data, 1)

        response = {'prediction': predicted.item()}
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'Error during prediction'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
