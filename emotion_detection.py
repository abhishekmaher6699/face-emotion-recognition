# from keras_facenet import FaceNet
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
import os
import cv2
import pickle
import tensorflow as tf

model = tf.keras.models.load_model("models/emotion-detector.h5")

# Initialize the MTCNN face detector
detector = MTCNN()

# Define the emotion labels corresponding to model's output
LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

# Extract the face region from the frame and preprocess it for emotion detection.
def get_face(face):

    # Get coordinates and dimensions of the face bounding box
    x1, y1, w, h = face['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + w, y1 + h

    # Extract the face region from the frame
    face = frame[y1:y2, x1:x2]

    # Convert the face region to grayscale
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Resize the face image to the required input size for the model
    face_image = Image.fromarray(face_gray)
    face_image = face_image.resize((64, 64))
    face_array = np.asarray(face_image)

    face_array = face_array.astype('float32') / 255.0

    # Reshape the face image to match the model's input shape
    face_array = face_array.reshape(1, 64, 64, 1)

    return face_array


# Predict the emotion from the face image using the emotion detection model.
def predictions(image):

    preds = model.predict(image)
    print(preds)
    pred = np.argmax(preds)
    emotion = LABELS[pred]
    return emotion


# Detect faces in the frame, predict their emotions, and annotate the frame with the results.
def detect_emotion(frame):

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)

    for face in faces:
        x, y, w, h = face['box']
        face_image = get_face(face)

        emotion = predictions(face_image)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame = detect_emotion(frame)
        cv2.imshow('Face Recognition', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()