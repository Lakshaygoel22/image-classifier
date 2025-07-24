from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)
model = load_model('image_classifier.h5')

# Load class names
with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img = load_img(file, target_size=(32, 32))
        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        class_id = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        return jsonify({
            'class_id': int(class_id),
            'class_name': class_names[class_id],
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
