import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os

class EmotionDetector:
    def __init__(self):
        # Emotion labels
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load pre-trained emotion detection model
        model_path = 'emotion_model.h5'
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            print("Model loaded successfully!")
        else:
            print("Warning: Model file not found. Please train or download a model.")
            self.model = None
    
    def detect_emotion(self, frame):
        """Detect faces and emotions in a frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if self.model is not None:
                # Extract face ROI
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                # Predict emotion
                prediction = self.model.predict(roi, verbose=0)[0]
                emotion_idx = np.argmax(prediction)
                emotion = self.emotion_labels[emotion_idx]
                confidence = prediction[emotion_idx]
                
                # Display emotion and confidence
                label = f"{emotion}: {confidence*100:.1f}%"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No model loaded", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        return frame
    
    def run(self):
        """Start webcam and detect emotions in real-time"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting emotion detection... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Detect emotions
            frame = self.detect_emotion(frame)
            
            # Display the frame
            cv2.imshow('Emotion Detection', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = EmotionDetector()
    detector.run()
