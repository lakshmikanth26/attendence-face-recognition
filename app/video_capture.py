import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load the labels from the labels.txt file
def load_labels(file_path):
    with open(file_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# Custom DepthwiseConv2D without 'groups'
def custom_depthwise_conv2d(*args, **kwargs):
    kwargs.pop('groups', None)  # Remove the 'groups' argument if it exists
    return tf.keras.layers.DepthwiseConv2D(*args, **kwargs)

# Load the model
model = tf.keras.models.load_model(
    'model/keras_model.h5',
    custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d}
)

# Load the labels
labels = load_labels('model/labels.txt')

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (MediaPipe works with RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # Preprocess the frame for prediction (resize and normalize as required by the model)
    resized_frame = cv2.resize(frame, (224, 224))  # Use the model's expected input size
    frame_array = np.expand_dims(resized_frame, axis=0) / 255.0  # Normalize if needed

    # Make predictions
    predictions = model.predict(frame_array)
    predicted_class = np.argmax(predictions)

    # Get the label corresponding to the predicted class
    predicted_label = labels[predicted_class]

    # Draw a bounding box around the hand(s) and display the label
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the bounding box of the hand based on the landmarks
            x_min = min([landmark.x for landmark in hand_landmarks.landmark])
            x_max = max([landmark.x for landmark in hand_landmarks.landmark])
            y_min = min([landmark.y for landmark in hand_landmarks.landmark])
            y_max = max([landmark.y for landmark in hand_landmarks.landmark])

            # Convert normalized coordinates to pixel values
            h, w, _ = frame.shape
            x_min, x_max = int(x_min * w), int(x_max * w)
            y_min, y_max = int(y_min * h), int(y_max * h)

            # Draw the bounding box around the hand
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Make the prediction label attractive (adjust font, background color, and positioning)
    label_text = f"Prediction: {predicted_label}"
    cv2.putText(frame, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Add a background box to make the text stand out
    (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (50, 50), (50 + w, 50 + h + 10), (0, 0, 0), -1)  # Black background
    cv2.putText(frame, label_text, (50, 50 + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
