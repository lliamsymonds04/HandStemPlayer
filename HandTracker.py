import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

from mediapipe import solutions

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class HandTracker:
    def __init__(self, model_path: str, camera_size_factor:int=0.6):
        self.model_path = model_path
        self.camera_size_factor = camera_size_factor
        self.screen_width, self.screen_height = pyautogui.size()
        self.start_time = time.time()
        self.cap = cv2.VideoCapture(0)

        # Initialize Mediapipe components
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.landmark_result = None

        # Setup the OpenCV window
        cv2.namedWindow("Hand Tracking")
        cv2.resizeWindow("Hand Tracking", int(self.screen_width * self.camera_size_factor),
                         int(self.screen_height * self.camera_size_factor))
        cv2.moveWindow("Hand Tracking", int(self.screen_width * (1 - self.camera_size_factor) * 0.5),
                       int(self.screen_height * (1 - self.camera_size_factor) * 0.5))

    def handle_detection(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.landmark_result = result

    def run(self):
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            result_callback=self.handle_detection,
            min_tracking_confidence=0.4,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )

        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print('Ignoring empty camera frame.')
                    break

                # Create a Mediapipe Image object
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame))


                # Call the async detection method
                now = time.time()
                landmarker.detect_async(mp_image, timestamp_ms=int((now - self.start_time) * 1000))

                # Convert the Mediapipe Image to NumPy for displaying
                mp_image_np = np.array(mp_image.numpy_view())

                height, width, depth = frame.shape
                if self.landmark_result is not None:

                    result: HandLandmarkerResult = self.landmark_result
                    for handedness in result.handedness:
                        cat = handedness[0]
                        hand = cat.display_name

                        if len(result.hand_landmarks) > 1:
                            landmarks = result.hand_landmarks[cat.index]
                        else:
                            landmarks = result.hand_landmarks[0]

                        for landmark in landmarks:
                            color = hand == "Right" and (0, 0, 255) or (0,255,0)
                            cv2.circle(mp_image_np, (int(landmark.x * width), int(landmark.y * height)), 10, color, cv2.FILLED)

                        # solutions.drawing_utils.draw_landmarks(
                        #     mp_image_np,
                        #     hand_landmarks,
                        #     solutions.hands.HAND_CONNECTIONS,
                        #     solutions.drawing_styles.get_default_hand_landmarks_style(),
                        #     solutions.drawing_styles.get_default_hand_connections_style())


                cv2.imshow("Hand Tracking", mp_image_np)

                # Exit on pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting...")
                    break

        self.cap.release()
        cv2.destroyAllWindows()