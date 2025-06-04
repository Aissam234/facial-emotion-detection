from flask import Flask, render_template, request, jsonify
import keras
import numpy as np
import cv2
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image
import base64
from io import BytesIO
from datetime import datetime


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
WEBCAM_FOLDER = 'static/webcam_captures'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['WEBCAM_FOLDER'] = WEBCAM_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WEBCAM_FOLDER, exist_ok=True)

# Dictionary to store our models
models = {}

# Load models with TensorFlow's Keras
try:
    models['v5'] = keras.models.load_model('model/emotion_model_v5.keras')
    models['fer2013'] = keras.models.load_model('model/emotion_model_fer2013.h5')
except Exception as e:
    print("Error loading models:", e)

emotion_labels = ['Angry', 'contempt', 'Disgusted', 'Fearful', 'Happy', 'neutral', 'sad', 'surprised']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_selected_model():
    model_name = request.args.get('model', 'v5')  # default to v5 if no model specified
    return models.get(model_name)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html')

    if 'image' not in request.files:
        return {'error': 'No file uploaded'}, 400

    model = get_selected_model()
    if not model:
        return {'error': 'Selected model not available'}, 400

    img_file = request.files['image']
    if not img_file:
        return {'error': 'Invalid file'}, 400

    # Save the uploaded file
    filename = secure_filename(img_file.filename)
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img_file.save(img_path)
    
    # Load and process image
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return render_template('upload.html', error="No face detected.")

    results = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (96, 96))
        face = face / 255.0
        face = face.reshape(1, 96, 96, 1)
        prediction = model.predict(face)[0]

        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        probabilities = {
            emotion_labels[i]: float(prediction[i]) 
            for i in range(len(emotion_labels))
        }

        # Draw rectangle on the image
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)
        
        face_result = {
            'emotion': emotion_labels[predicted_class],
            'confidence': float(confidence),
            'probabilities': probabilities,
            'face_rect': {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            }
        }
        results.append(face_result)
    
    # Save the annotated image
    cv2.imwrite(img_path, img)

    # If it's an AJAX request, return JSON with all faces
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return {'results': results, 'image_path': img_path}
    
    # Otherwise return the template with all faces data
    return render_template(
        'upload.html',
        results=results,
        image_path=img_path
    )



@app.route('/webcam')
def webcam():
    return render_template('webcam_new.html')

@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    try:
        model = get_selected_model()
        if not model:
            return jsonify({'error': 'Selected model not available'}), 500

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data received'}), 400

        # Process the image data
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)

        # Save image to webcam_captures folder with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'webcam_{timestamp}.jpg'
        
        # Ensure the directory exists
        os.makedirs('static/webcam_captures', exist_ok=True)
        
        # Save the image
        img_path = os.path.join('static/webcam_captures', filename)
        with open(img_path, 'wb') as f:
            f.write(img_bytes)
        
        # Save image to webcam_captures folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'webcam_{timestamp}.jpg'
        img_path = os.path.join('static/webcam_captures', filename)
        
        # Ensure directory exists
        os.makedirs('static/webcam_captures', exist_ok=True)
        
        # Save the image
        with open(img_path, 'wb') as f:
            f.write(img_bytes)

        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Return early if no faces detected
        if len(faces) == 0:
            return jsonify({
                'results': [],
                'message': 'No faces detected. Please ensure your face is well-lit and centered in the frame.'
            })

        # Process detected faces
        faces_data = []
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (96, 96))
            face = face / 255.0
            face = face.reshape(1, 96, 96, 1)
            prediction = model.predict(face)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]

            probabilities = {
                emotion_labels[i]: float(prediction[i]) 
                for i in range(len(emotion_labels))
            }
            
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_data = {
                'emotion': emotion_labels[predicted_class],
                'confidence': float(confidence),
                'face_rect': {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                },
                'probabilities': probabilities
            }
            faces_data.append(face_data)            # Save the annotated image
            cv2.imwrite(img_path, img)
            
            return jsonify({
                'results': faces_data,
                'image': f'/static/webcam_captures/{filename}'
            })

    except Exception as e:
        print(f"Error in predict_webcam: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)