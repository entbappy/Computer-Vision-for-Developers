import os
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your trained model
MODEL_PATH = 'model/model_vgg16.h5'
model = load_model(MODEL_PATH)

# Replace with your actual class names in order of model outputs
class_names = [
    'Oblique fracture',
    'Spiral Fracture'
]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # preprocess
            img = load_img(filepath, target_size=(224, 224))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # predict
            preds = model.predict(x)
            idx = np.argmax(preds[0])
            label = class_names[idx]
            confidence = preds[0][idx]

            return render_template('index.html',
                                   filename=file.filename,
                                   label=label,
                                   confidence=f"{confidence*100:.2f}%")
        return redirect(request.url)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080,debug=True)
