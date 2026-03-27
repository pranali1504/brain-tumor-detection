from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model/tumor_model.h5')

# Tumor class labels
classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

@app.route('/')
def index():
    return render_template('index.html', label=None)

@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded."

    img_file = request.files['image']
    if img_file.filename == '':
        return "No selected file."

    filepath = os.path.join('static/uploads', img_file.filename)
    img_file.save(filepath)

    # Load and preprocess image
    img = image.load_img(filepath, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    predicted_label = classes[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    result = f"{predicted_label} ({confidence:.2f}% confidence)"
    return render_template('index.html', label=result, confidence=f"{confidence:.2f}",image_path='/'+filepath)

if __name__ == '__main__':
    app.run(debug=True)
