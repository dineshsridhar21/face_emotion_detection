import tensorflow as tf
import face_recognition
import cv2
import pickle
import numpy as np
import os
import time

# Face Image Directory
directory = r"C:\Users\Dinesh\Office_work\face_emotion_detection\images"
known_face_encodings = [] #encoding for face
known_face_labels = [] #Labels for person

def loop_through_dir(directory):

    # Loop through to directories
    for root, dirs, files in os.walk(directory):
        for file in files:

            # checking particular file image format
            if file.endswith(('png', 'jpg', 'jpeg')):
                file_path = os.path.join(root, file)
                label = os.path.basename(root)
                image = face_recognition.load_image_file(file_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_labels.append(label)
                    print(f"Loop throughed in this directory {file_path} labeled as an {label}")

loop_through_dir(directory)

# Ensure the directory exists before saving the file
output_directory = r"C:\Users\Dinesh\Office_work\face_emotion_detection"
os.makedirs(output_directory, exist_ok=True)

# Creating face trained dataset
with open(os.path.join(output_directory, "known_faces.pkl"), "wb") as f:
    pickle.dump((known_face_encodings, known_face_labels), f)

# Load dataset for Face
with open(os.path.join(output_directory, "known_faces.pkl"), "rb") as f:
    known_face_encodings, known_face_labels = pickle.load(f)

#Load dataset for Emotion
emotion_model = tf.keras.models.load_model('emotion_model_weights.h5')

emotion_labels = {1: 'Angry', 2: 'Disgust', 3: 'Fear', 4: 'Happy', 5: 'Sad', 6: 'Surprise', 7: 'Neutral'}
emotion_tracking = {}

def face_emotion(img):
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Detecting Face Location
        top, right, bottom, left = face_location

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)  # compare faces with encodings
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        # Checking for best faces
        if matches[best_match_index]:
            predicted_face = known_face_labels[best_match_index]
        else:
            # If no images provided, label as unknown
            predicted_face = "Unknown"

        face_roi = img[top:bottom, left:right]
        resized_emotion = cv2.resize(face_roi, (48, 48))
        resized_emotion_gray = cv2.cvtColor(resized_emotion, cv2.COLOR_BGR2GRAY)
        normalized_emotion = resized_emotion_gray / 255.0
        reshaped_emotion = np.reshape(normalized_emotion, (1, 48, 48, 1))

        emotion_prediction = emotion_model.predict(reshaped_emotion)
        predicted_emotion = emotion_labels[np.argmax(emotion_prediction)]

        face_id = f"{left}-{top}-{right}-{bottom}"
        current_time = time.time()

        # Calculation Emotion timing spend by the person
        if face_id not in emotion_tracking:
            emotion_tracking[face_id] = {
                'emotion': predicted_emotion,
                'start_time': current_time,
                'elapsed_time': 0
            }
        else:
            if emotion_tracking[face_id]['emotion'] == predicted_emotion:
                elapsed_time = current_time - emotion_tracking[face_id]['start_time']
                emotion_tracking[face_id]['elapsed_time'] = elapsed_time
            else:
                emotion_tracking[face_id] = {
                    'emotion': predicted_emotion,
                    'start_time': current_time,
                    'elapsed_time': 0
                }

        processing_time = emotion_tracking[face_id]['elapsed_time']

        # Alert message spending over same emotion
        alert_message = ""
        if processing_time > 5.00:
            alert_message = "ALERT: Detected for over 5 seconds!"

        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

        # Providing a font style for  Face, Emotion, Time spent on the Emotion
        cv2.putText(img, f"Face: {predicted_face}", (left + 5, top - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255),
                    2)
        cv2.putText(img, f"Emotion: {predicted_emotion}", (left + 5, top - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        cv2.putText(img, f"Time spent: {processing_time:.2f}s", (left + 5, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        if alert_message:
            cv2.putText(img, f"{alert_message}", (left + 5, bottom + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return img

# Capturing the Video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    processed_frame = face_emotion(frame)

    cv2.imshow("Face and Emotion Detection", processed_frame)

    # Press 'q' for stop the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
