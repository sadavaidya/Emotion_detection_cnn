import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2


# Load the saved model
model = tf.keras.models.load_model('emotion_detection_model.h5')


# Load and preprocess the image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(48, 48), color_mode="grayscale")
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

# Predict emotion on a new image
# img_path = 'D:\\ML_Projects\\Dataset\\test\\fear\\PrivateTest_134207.jpg'
# new_img = load_and_preprocess_image(img_path)
# pred = model.predict(new_img)

# Get the predicted label
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# predicted_emotion = emotion_labels[np.argmax(pred)]
# print("Predicted Emotion:", predicted_emotion)

def real_time_emotion_detection():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            # Extract the region of interest (face)
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)
            
            # Make prediction on the face
            prediction = model.predict(roi_gray)
            predicted_emotion = emotion_labels[np.argmax(prediction)]
            
            # Draw a rectangle around the face and display the predicted emotion
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show the frame with the emotion label
        cv2.imshow('Emotion Detector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Run real-time emotion detection
real_time_emotion_detection()
