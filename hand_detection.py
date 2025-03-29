import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

previous_time = 0
current_time = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    hand_count = 0
    
    if results.multi_hand_landmarks:
        hand_count = len(results.multi_hand_landmarks)
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(
                                    img, 
                                    handLms, 
                                    mpHands.HAND_CONNECTIONS, 
                                    mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),  # Red dots
                                    mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2) # Green Lines
                                 )
            
            
    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time
            
    img = cv2.flip(img, 1)
    cv2.putText(img, f"Hands Detected: {hand_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"FPS: {int(fps)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
