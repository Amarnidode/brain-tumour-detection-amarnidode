from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from predict import predict_tumor

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')

# Class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to predict tumor type


# Route for the main page (index.html)
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

# Route for the contact page (contact.html)
@app.route('/contact')
def contact():
    return render_template('contact.html')


# Route for the detect page (detect.html)
@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        if file:
            # Save the file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            # Predict the tumor
            result, confidence = predict_tumor(file_location)

            # Return result along with image path for display
            return render_template('detect.html', result=result, confidence=f"{confidence*100:.2f}%", file_path=f'/uploads/{file.filename}')

    return render_template('detect.html', result=None)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
