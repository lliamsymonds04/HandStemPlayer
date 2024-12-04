from typing import Tuple, Any

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

from sympy.physics.vector import frame

from PointUtil import *

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

#angle tolerances
INDEX_FINGER_MAX_ANGLE = 80
INDEX_FINGER_MIN_ANGLE = 25
MIDDLE_FINGER_MAX_ANGLE = 80
MIDDLE_FINGER_MIN_ANGLE = 25

#Circle Sizes
MAX_CIRCLE_SIZE = 10
MIN_CIRCLE_SIZE = 2


def draw_line_between_points(point1: NormalizedLandmark, point2: NormalizedLandmark, frame: cv2.Mat, image: np.ndarray[Any, np.dtype], color: Tuple[int, int, int], thickness: int):
    height, width, depth = frame.shape

    p1 = (int(point1.x * width), int(point1.y * height))
    p2 = (int(point2.x * width), int(point2.y * height))
    cv2.line(image, p1, p2, color, thickness)


class HandTracker:
    def __init__(self, model_path: str, camera_size_factor:float=0.6):
        self.model_path = model_path
        self.camera_size_factor = camera_size_factor
        self.screen_width, self.screen_height = pyautogui.size()
        self.start_time = time.time()
        self.cap = cv2.VideoCapture(0)

        # Initialize Mediapipe components
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.landmark_result = None

        # Set up the finger variables
        self.left_index_finger_closure = 0
        self.right_index_finger_closure = 0
        self.left_middle_finger_closure = 0
        self.right_middle_finger_closure = 0
        self.distance_between_thumbs = 1

        self.looping = False

        # Set up the OpenCV window
        cv2.namedWindow("Hand Stem Player")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.screen_width * self.camera_size_factor))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.screen_height * self.camera_size_factor))

        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # cv2.resizeWindow("Hand Stem Player", int(self.screen_width * self.camera_size_factor),
        #                  int(self.screen_height * self.camera_size_factor))
        # cv2.moveWindow("Hand Stem Player", int((self.screen_width - frame_width) * 0.5),
        #                max(int((self.screen_height - frame_height) * 0.25),0))

        cv2.resizeWindow("Hand Stem Player", frame_width, frame_height)
        # cv2.imshow("Hand Stem Player", frame)


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
                    self.left_index_finger_closure = index_finger_closure
                    self.left_middle_finger_closure = middle_finger_closure

                #visual stuff
                index_finger_colour = value_to_color(index_finger_closure, "winter")
                middle_finger_colour = value_to_color(middle_finger_closure, "summer")

                index_finger_circle_size = int(lerp(MIN_CIRCLE_SIZE, MAX_CIRCLE_SIZE, index_finger_closure))
                middle_finger_circle_size = int(lerp(MIN_CIRCLE_SIZE, MAX_CIRCLE_SIZE, middle_finger_closure))

                #drawing
                if index_finger_closure > 0:
                    cv2.circle(mp_image_np, (int(index_tip.x * width), int(index_tip.y * height)),
                               index_finger_circle_size, index_finger_colour, cv2.FILLED)
                    draw_line_between_points(thumb_tip, index_tip, frame, mp_image_np, index_finger_colour, 5)

                if middle_finger_closure > 0:
                    cv2.circle(mp_image_np, (int(middle_tip.x * width), int(middle_tip.y * height)),
                               middle_finger_circle_size, middle_finger_colour, cv2.FILLED)
                    draw_line_between_points(thumb_tip, middle_tip, frame, mp_image_np, middle_finger_colour, 5)

            if len(result.hand_landmarks) == 2:
                mark1 = result.hand_landmarks[0]
                mark2 = result.hand_landmarks[1]

                thumb1 = mark1[4]
                thumb2 = mark2[4]

                self.distance_between_thumbs = get_distance_between_points(thumb1, thumb2)

        cv2.imshow("Hand Stem Player", mp_image_np)


