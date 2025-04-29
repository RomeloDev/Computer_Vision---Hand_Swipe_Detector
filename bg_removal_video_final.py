import cv2  # OpenCV for image and video processing
import cvzone  # CVZone for easier computer vision tasks
from cvzone.SelfiSegmentationModule import SelfiSegmentation  # For removing background using segmentation
import mediapipe as mp  # MediaPipe for hand tracking
import time  # To handle timing and FPS
import os  # To access the filesystem

# Class to detect swipe gestures based on hand movement
class SwipeDetector:
    def __init__(self, threshold=0.20, display_duration=1):
        # Initialize MediaPipe Hands
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils  # Drawing utilities for landmarks
        self.previous_x = None  # Previous X position of the hand
        self.current_x = None  # Current X position of the hand
        self.threshold = threshold  # Minimum X movement to consider it a swipe
        self.swipe_direction = None  # Detected swipe direction: "left" or "right"
        self.last_swipe_time = 0  # Time when last swipe was detected
        self.display_duration = display_duration  # How long a swipe should remain active

    # Determines swipe direction based on X-axis movement
    def detect_swipe(self, current_x):
        self.current_x = current_x
        if self.previous_x is not None:
            diff = self.current_x - self.previous_x
            if abs(diff) > self.threshold:
                self.swipe_direction = "right" if diff > 0 else "left"
                print(diff)  # Debugging print to show swipe difference
                self.last_swipe_time = time.time()  # Record swipe time
        self.previous_x = self.current_x

    # Processes each frame to detect hand landmarks
    def process_frame(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB
        results = self.hands.process(img_rgb)  # Get hand landmark results

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get X position of wrist to detect swipe
                current_x = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x
                self.detect_swipe(current_x)

                # Uncomment below to draw hand landmarks for visualization
                # self.mp_draw.draw_landmarks(
                #     img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                #     self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                #     self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                # )

        return img

    # Resets tracking values for new swipe
    def reset_tracking(self):
        self.current_x = None
        self.previous_x = None
        self.swipe_direction = None

# Loads all background videos from a folder
def load_background_videos(folder_path):
    bg_videos = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.mp4'):
            path = os.path.join(folder_path, file)
            bg_videos.append(cv2.VideoCapture(path))  # Add video to list
    return bg_videos

# Resets the background video to the beginning
def reset_video(video):
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Main program logic
def main():
    cap = cv2.VideoCapture(0)  # Start webcam
    cap.set(3, 1920)  # Set width
    cap.set(4, 1080)  # Set height

    segmentator = SelfiSegmentation()  # Initialize segmentation
    swipe_detector = SwipeDetector()  # Initialize swipe detector

    bg_videos = load_background_videos('bg_videos')  # Load background videos
    current_bg_index = -1  # -1 means use the original background

    previous_time = time.time()  # Used for FPS calculation

    window_name = 'Virtual Background'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        success, img = cap.read()
        if not success:
            print("Webcam not found.")
            break

        img = cv2.flip(img, 1)  # Mirror the image for natural interaction
        img = cv2.resize(img, (1200, 760))  # Resize for display

        img = swipe_detector.process_frame(img)  # Process for hand swipes

        # Handle swipe gestures to change background video
        if swipe_detector.swipe_direction:
            if time.time() - swipe_detector.last_swipe_time < swipe_detector.display_duration:
                if swipe_detector.swipe_direction == "right":
                    current_bg_index += 1
                    if current_bg_index > len(bg_videos) - 1:
                        current_bg_index = -1  # Go back to original
                elif swipe_detector.swipe_direction == "left":
                    current_bg_index -= 1
                    if current_bg_index < -1:
                        current_bg_index = len(bg_videos) - 1  # Wrap to last video
                swipe_detector.reset_tracking()  # Clear direction after processing
                print(f'current_x: {swipe_detector.current_x}')  # Debugging info

        # Determine which background to show
        if current_bg_index == -1:
            imgOut = img
            cv2.putText(imgOut, f"Current Background: Original", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            success_bg, imgBg = bg_videos[current_bg_index].read()
            if not success_bg:
                reset_video(bg_videos[current_bg_index])
                success_bg, imgBg = bg_videos[current_bg_index].read()

            imgBg = cv2.resize(imgBg, (img.shape[1], img.shape[0]))  # Resize background to match webcam
            imgOut = segmentator.removeBG(img, imgBg, 0.50)  # Remove original background and overlay new
            cv2.putText(imgOut, f"Current Background: {current_bg_index + 1}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(imgOut, f"FPS: {int(fps)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Virtual Background', imgOut)  # Show output image

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q' press
            break

    cap.release()
    for video in bg_videos:
        video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
