import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import mediapipe as mp
import time
import os

class SwipeDetector:
    def __init__(self, threshold=0.10, display_duration=1):
        self.hands = mp.solutions.hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils
        self.previous_x = None
        self.threshold = threshold
        self.swipe_direction = None
        self.last_swipe_time = 0
        self.display_duration = display_duration

    def detect_swipe(self, current_x):
        if self.previous_x is not None:
            diff = current_x - self.previous_x
            if abs(diff) > self.threshold:
                self.swipe_direction = "right" if diff > 0 else "left"
                self.last_swipe_time = time.time()
        self.previous_x = current_x

    def process_frame(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                current_x = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x
                self.detect_swipe(current_x)
                self.mp_draw.draw_landmarks(
                    img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )
        
        return img

def load_background_videos(folder_path):
    bg_videos = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.mp4'):
            path = os.path.join(folder_path, file)
            bg_videos.append(cv2.VideoCapture(path))
    return bg_videos

def reset_video(video):
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    segmentator = SelfiSegmentation()
    swipe_detector = SwipeDetector()

    bg_videos = load_background_videos('bg_videos')
    current_bg_index = -1  # -1 means original background (no video)

    previous_time = time.time()

    while True:
        success, img = cap.read()
        if not success:
            print("Webcam not found.")
            break

        img = cv2.flip(img, 1)
        img = cv2.resize(img, (1200, 800))

        img = swipe_detector.process_frame(img)

        # Handle swipe events
        if swipe_detector.swipe_direction:
            if time.time() - swipe_detector.last_swipe_time < swipe_detector.display_duration:
                if swipe_detector.swipe_direction == "right":
                    current_bg_index = (current_bg_index + 1) % len(bg_videos)
                elif swipe_detector.swipe_direction == "left":
                    current_bg_index = (current_bg_index - 1) % len(bg_videos)
                swipe_detector.swipe_direction = None  # Reset after processing

        # Set background
        if current_bg_index == -1:
            imgOut = img
        else:
            success_bg, imgBg = bg_videos[current_bg_index].read()
            if not success_bg:
                reset_video(bg_videos[current_bg_index])
                success_bg, imgBg = bg_videos[current_bg_index].read()

            # imgBg = cv2.resize(imgBg, (img.shape[1], img.shape[0]))
            imgBg = cv2.resize(imgBg, (1200, 800))
            imgOut = segmentator.removeBG(img, imgBg, 0.90)

        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(imgOut, f"FPS: {int(fps)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Virtual Background', imgOut)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    for video in bg_videos:
        video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
