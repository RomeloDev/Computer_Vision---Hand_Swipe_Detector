import cv2
import cvzone
import os
import time
import mediapipe as mp
from cvzone.SelfiSegmentationModule import SelfiSegmentation

# Initialize
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
segmentator = SelfiSegmentation()
previous_time = 0

# Load background images
bg_folder = 'assets'
bg_files = sorted(os.listdir(bg_folder))  # Sort to maintain order
bg_images = [cv2.imread(os.path.join(bg_folder, bg)) for bg in bg_files]

# Swipe Detector Initialization
hands = mp.solutions.hands.Hands()
mp_draw = mp.solutions.drawing_utils
previous_x = None
last_swipe_time = 0
swipe_text = ""
display_duration = 1
threshold = 0.10

current_bg_index = -1  # -1 means original webcam without background

def detect_swipe(current_x, previous_x):
    global last_swipe_time, swipe_text, current_bg_index
    if previous_x is not None:
        diff = current_x - previous_x
        if abs(diff) > threshold:
            if diff > 0:
                swipe_text = "Swipe Right"
                current_bg_index += 1
            else:
                swipe_text = "Swipe Left"
                current_bg_index -= 1
            last_swipe_time = time.time()
            
            # Clamp the index to valid range
            if current_bg_index < -1:
                current_bg_index = -1
            elif current_bg_index >= len(bg_images):
                current_bg_index = len(bg_images) - 1

    return current_x

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    # Process Hand Detection
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            current_x = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x
            previous_x = detect_swipe(current_x, previous_x)
            mp_draw.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    # Background Replacement
    if current_bg_index == -1:
        imgOut = img  # Show original
    else:
        imgOut = segmentator.removeBG(img, bg_images[current_bg_index], 0.5)

    # Stack Images
    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    # Display Swipe Info
    if time.time() - last_swipe_time < display_duration:
        cv2.putText(imgStacked, swipe_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display Other Info
    cv2.putText(imgStacked, f"Background: {current_bg_index if current_bg_index != -1 else 'Original'}", 
                (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(imgStacked, f"FPS: {int(fps)}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the result
    cv2.imshow("BG Remover + Swipe Controller", imgStacked)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()
