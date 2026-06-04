# 🎭 Facial Emotion Detection Web App

<div align="center">
  <p align="center">
    A lightweight, robust Flask-based web application that detects human emotions from facial expressions using deep learning and OpenCV.
  </p>
</div>

## 📖 About The Project

This project leverages the power of Convolutional Neural Networks (CNNs) and computer vision to analyze human faces and predict emotions in real-time. Built with **Flask**, **TensorFlow/Keras**, and **OpenCV**, the application provides a user-friendly interface to either upload static images or use a live webcam feed for instant emotion analysis.

The system detects faces using Haar Cascades and classifies the dominant emotion into one of eight categories:
*Angry, Contempt, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.*

### ✨ Key Features

- **Dual Input Modes:** Upload static images or use your live webcam feed for real-time predictions.
- **Deep Learning Models:** Supports multiple pre-trained Keras models (`v5` and `fer2013`) for flexible and accurate predictions.
- **Face Detection:** Accurately isolates faces in the frame using OpenCV's Haarcascade before performing inference.
- **Confidence Scoring:** Provides visual bounding boxes along with prediction confidence and probability distributions for all emotion classes.
- **Interactive UI:** A clean, responsive web interface built with HTML/CSS and AJAX for seamless interactions.

### 🧠 Built With

* [Flask](https://flask.palletsprojects.com/)
* [TensorFlow & Keras](https://www.tensorflow.org/)
* [OpenCV](https://opencv.org/)
* [NumPy](https://numpy.org/)
* [Pillow](https://python-pillow.org/)

---

## 🚀 Getting Started

Follow these steps to run the application locally.

### Prerequisites

Ensure you have Python 3.x installed on your machine. 

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Aissam234/facial-emotion-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd facial-emotion-detection
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 Usage

Start the Flask development server:

```bash
python app.py
```

Once the server is running, open your web browser and navigate to:
`http://localhost:8000`

### Application Modes

1. **Upload Image:** Navigate to the upload section, select an image containing one or more faces, and hit submit. The app will draw bounding boxes and display the predicted emotion for each face.
2. **Live Webcam:** Navigate to the webcam section to grant browser camera permissions. The app will capture frames, send them to the backend asynchronously, and display live bounding boxes and emotion predictions.

---

## 📂 Project Structure

- **`app.py`**: The core Flask application managing routes, image processing, face detection, and model inference.
- **`model/`**: Contains the pre-trained deep learning models (`.keras` and `.h5` formats).
- **`templates/`**: HTML templates for the web interface (`home.html`, `upload.html`, `webcam_new.html`).
- **`static/`**: Holds uploaded images, webcam captures, and potentially CSS/JS assets.
- **`DL_project_v4.ipynb` & `fer_version.ipynb`**: Jupyter Notebooks containing the data exploration, preprocessing, and training pipelines for the deep learning models.
- **`Procfile`**: Configuration for deployment (e.g., to Heroku).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
