import cv2
import mediapipe as mp
import time

class SwipeDetector:
    def __init__(self, threshold=0.10, display_duration=1):
        self.cap = cv2.VideoCapture(0) # Open the webcam
        self.hands = mp.solutions.hands.Hands() # Initialize mediapipe hands model
        self.mp_draw = mp.solutions.drawing_utils # Utility for drawing hand landmarks
        self.previous_x = None
        self.threshold = threshold
        self.swipe_text = ""
        self.last_swipe_time = 0
        self.display_duration = display_duration
        self.previous_time = 0

    def detect_swipe(self, current_x):
        if self.previous_x is not None:
            diff = current_x - self.previous_x
            if abs(diff) > self.threshold:
                self.swipe_text = "Right Swipe Detected" if diff > 0 else "Left Swipe Detected"
                self.last_swipe_time = time.time()
        self.previous_x = current_x

    def process_frame(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        hand_count = 0

        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            for hand_landmarks in results.multi_hand_landmarks:
                current_x = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x
                self.detect_swipe(current_x)
                self.mp_draw.draw_landmarks(
                    img, 
                    hand_landmarks, 
                    mp.solutions.hands.HAND_CONNECTIONS, 
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3), 
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )
        
        return img, hand_count

    def display_info(self, img, hand_count):
        current_time = time.time()
        fps = 1 / (current_time - self.previous_time)
        self.previous_time = current_time

        if time.time() - self.last_swipe_time < self.display_duration:
            cv2.putText(img, self.swipe_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(img, f"Hands Detected: {hand_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"FPS: {int(fps)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img

    def run(self):
        while True:
            success, img = self.cap.read()
            if not success:
                break
            img = cv2.flip(img, 1)
            img, hand_count = self.process_frame(img)
            img = self.display_info(img, hand_count)
            cv2.imshow("Swipe Detector", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = SwipeDetector()
    detector.run()
