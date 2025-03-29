import cv2
import mediapipe as mp
import time

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Variables to track hand position and swipe display duration
previous_time = 0
previous_x = None
threshold = 0.10 
swipe_text = ""
last_swipe_time = 0
display_duration = 1  # Seconds to display swipe text

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Flip the image for a mirror effect
    img = cv2.flip(img, 1)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    hand_count = 0

    if results.multi_hand_landmarks:
        hand_count = len(results.multi_hand_landmarks)
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the x-coordinate of the wrist
            current_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x

            # Check for swipe gesture
            if previous_x is not None:
                diff = current_x - previous_x
                if abs(diff) > threshold:
                    if diff > 0:
                        swipe_text = "Right Swipe Detected"
                    else:
                        swipe_text = "Left Swipe Detected"
                    last_swipe_time = time.time()  # Reset timer

            previous_x = current_x

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),  # Red dots
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)  # Green lines
            )

    # Display swipe text only if within display duration
    if time.time() - last_swipe_time < display_duration:
        cv2.putText(img, swipe_text, (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Calculate and display FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time


    # Display hand count and FPS
    cv2.putText(img, f"Hands Detected: {hand_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"FPS: {int(fps)}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the image
    cv2.imshow("Image", img)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()