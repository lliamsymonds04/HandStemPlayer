import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

from PointUtil import *

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

INDEX_FINGER_MAX_ANGLE = 80
INDEX_FINGER_MIN_ANGLE = 25
MIDDLE_FINGER_MAX_ANGLE = 80
MIDDLE_FINGER_MIN_ANGLE = 25

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

        # Setup the finger variables
        self.left_index_finger_closure = 0
        self.right_index_finger_closure = 0
        self.left_middle_finger_closure = 0
        self.right_middle_finger_closure = 0

        # Setup the OpenCV window
        cv2.namedWindow("Hand Tracking")
        cv2.resizeWindow("Hand Tracking", int(self.screen_width * self.camera_size_factor),
                         int(self.screen_height * self.camera_size_factor))
        cv2.moveWindow("Hand Tracking", int(self.screen_width * (1 - self.camera_size_factor) * 0.5),
                       int(self.screen_height * (1 - self.camera_size_factor) * 0.5))

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            result_callback=self.handle_detection,
            min_tracking_confidence=0.4,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )

        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)


    def handle_detection(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.landmark_result = result

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            print('Ignoring empty camera frame.')
            return

        # Create a Mediapipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame))

        # Call the async detection method
        now = time.time()
        self.landmarker.detect_async(mp_image, timestamp_ms=int((now - self.start_time) * 1000))

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
                    color = hand == "Right" and (0, 0, 255) or (0, 255, 0)
                    cv2.circle(mp_image_np, (int(landmark.x * width), int(landmark.y * height)), 10, color, cv2.FILLED)

                # joints of importance
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                index_knuckle = landmarks[5]
                middle_knuckle = landmarks[9]

                # calc angles
                index_finger_angle = get_angle_between_points(thumb_tip, index_tip, index_knuckle)
                middle_finger_angle = get_angle_between_points(thumb_tip, middle_tip, middle_knuckle)

                # clamp the angles
                index_finger_closure = (max(min(index_finger_angle, INDEX_FINGER_MAX_ANGLE), INDEX_FINGER_MIN_ANGLE) - INDEX_FINGER_MIN_ANGLE) / (
                            INDEX_FINGER_MAX_ANGLE - INDEX_FINGER_MIN_ANGLE)
                middle_finger_closure = (max(min(middle_finger_angle, MIDDLE_FINGER_MAX_ANGLE), MIDDLE_FINGER_MIN_ANGLE) - MIDDLE_FINGER_MIN_ANGLE) / (
                                                    MIDDLE_FINGER_MAX_ANGLE - MIDDLE_FINGER_MIN_ANGLE)

                #the handles are flipped so do opposite
                if hand == "Left":
                    self.right_index_finger_closure = index_finger_closure
                    self.right_middle_finger_closure = middle_finger_closure
                elif hand == "Right":
                    # print(index_finger_angle)
                    self.left_index_finger_closure = index_finger_closure
                    self.left_middle_finger_closure = middle_finger_closure

        cv2.imshow("Hand Tracking", mp_image_np)