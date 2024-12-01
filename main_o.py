import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

from mediapipe import solutions

from PointUtil import get_distance, convert_to_pixels, value_to_color, get_mid_point, lerp

# Initialize Mediapipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open webcam
screen_width, screen_height = pyautogui.size()
camera_size_factor = 0.6

cv2.namedWindow("Hand Tracking")
cv2.resizeWindow("Hand Tracking", int(screen_width * camera_size_factor), int(screen_height * camera_size_factor))
cv2.moveWindow("Hand Tracking", int(screen_width * (1 - camera_size_factor) * 0.5),
               int(screen_height * (1 - camera_size_factor) * 0.5))

cap = cv2.VideoCapture(0)

MIN_CIRCLE_SIZE = 2
MAX_CIRCLE_SIZE = 15

MID_POINT_MAX_LEN = 0.35
PINCH_MAX_LEN = 0.25

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

landmark_result: HandLandmarkerResult


def handle_detection(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    landmark_result = result


# Create a hand landmarker instance with the video mode:
landmark_model_path = 'Models\\hand_landmarker.task'
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=landmark_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handle_detection
)

# https://stackoverflow.com/questions/76533527/why-isnt-mediapipe-drawing-the-landmarks-on-the-live-feed
start_time = time.time()
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Ignoring empty camera frame.')
            break

        # Create a Mediapipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Call the async detection method
        now = time.time()
        timestamp = int((now - start_time) * 1000)
        landmarker.detect_async(mp_image, timestamp_ms=timestamp)

        # handle drawing
        landmarker.detect_async(mp_image, timestamp_ms=timestamp)

        # Convert the Mediapipe Image to NumPy for displaying
        mp_image_np = np.array(mp_image.numpy_view())
        # cv2.imshow('Hand Tracking', cv2.cvtColor(mp_image_np, cv2.COLOR_RGB2BGR))
        cv2.imshow("Hand Tracking", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("break")
            break

states = ["Calibrating"]
state = 0

# with mp_hands.Hands(
#         static_image_mode=False,  # Real-time detection
#         max_num_hands=2,  # Detect up to 2 hands
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5) as hands:
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # Display the frame
#

#
#         # # Flip the image horizontally for a mirrored effect
#         # frame = cv2.flip(frame, 1)
#
#         # Convert the image to RGB (Mediapipe uses RGB format)
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # Process the frame and detect hands
#         results = hands.process(rgb_frame)
#
#         height, width, depth = frame.shape
#
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # Draw landmarks and connections on the original frame
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, )
#
#             if state == 0:
#                 pass
#
#             # if len(results.multi_hand_landmarks) == 2:
#             #     index_0 = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#             #     thumb_0 = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP]
#             #     index_1 = results.multi_hand_landmarks[1].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#             #     thumb_1 = results.multi_hand_landmarks[1].landmark[mp_hands.HandLandmark.THUMB_TIP]
#
#             #     index_0_p = convert_to_pixels(index_0, frame)
#             #     thumb_0_p = convert_to_pixels(thumb_0, frame)
#             #     index_1_p = convert_to_pixels(index_1, frame)
#             #     thumb_1_p = convert_to_pixels(thumb_1, frame)
#
#             #     mid_point_0 = get_mid_point(index_0_p, thumb_0_p)
#             #     mid_point_1 = get_mid_point(index_1_p, thumb_1_p)
#
#             #     print(index_0.z)
#
#             #     # dist = get_distance(index_0, index_1)
#
#             #     # text_pos = (int(index_0.x * width), int(index_0.y * height))
#             #     # cv2.putText(
#             #     #     frame,
#             #     #     str(dist),
#             #     #     text_pos,
#             #     #     cv2.FONT_HERSHEY_SIMPLEX,  # Font type
#             #     #     1,                         # Font scale
#             #     #     (0, 255, 0),               # Text color (green)
#             #     #     2,                         # Thickness
#             #     #     cv2.LINE_AA
#
#             #     # )
#
#             #     #middle line
#             #     mid_point_dist = get_distance(mid_point_0, mid_point_1)
#
#             #     mid_line_float = min(mid_point_dist/(width * MID_POINT_MAX_LEN), 1)
#             #     mid_point_colour = value_to_color(mid_line_float, "cool")
#             #     mid_circle_size = int(lerp(MIN_CIRCLE_SIZE, MAX_CIRCLE_SIZE, mid_line_float))
#
#             #     cv2.circle(frame, mid_point_0, mid_circle_size, mid_point_colour, cv2.FILLED)
#             #     cv2.circle(frame, mid_point_1, mid_circle_size, mid_point_colour, cv2.FILLED)
#             #     cv2.line(frame, mid_point_0, mid_point_1, mid_point_colour, 3)
#
#             #     #left line
#             #     left_point_dist = get_distance(index_0_p, thumb_0_p)
#
#             #     left_line_float = min(left_point_dist/(width * PINCH_MAX_LEN), 1)
#             #     left_point_colour = value_to_color(left_line_float, "hot")
#             #     left_circle_size = int(lerp(MIN_CIRCLE_SIZE, MAX_CIRCLE_SIZE, left_line_float))
#
#             #     cv2.circle(frame, index_0_p, left_circle_size, left_point_colour, cv2.FILLED)
#             #     cv2.circle(frame, thumb_0_p, left_circle_size, left_point_colour, cv2.FILLED)
#             #     cv2.line(frame, index_0_p, thumb_0_p, left_point_colour, 3)
#
#             #     #right line
#             #     right_point_dist = get_distance(index_1_p, thumb_1_p)
#
#             #     right_line_float = min(right_point_dist/(width * PINCH_MAX_LEN), 1)
#             #     right_point_colour = value_to_color(right_line_float, "spring")
#             #     right_circle_size = int(lerp(MIN_CIRCLE_SIZE, MAX_CIRCLE_SIZE, right_line_float))
#
#             #     cv2.circle(frame, index_1_p, right_circle_size, right_point_colour, cv2.FILLED)
#             #     cv2.circle(frame, thumb_1_p, right_circle_size, right_point_colour, cv2.FILLED)
#             #     cv2.line(frame, index_1_p, thumb_1_p, right_point_colour, 3)
#
#         # Display the frame
#         cv2.imshow('Hand Tracking', frame)
#
#         # Exit on pressing 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# print(hand_landmarks)
# landmarks = [(lm.x, lm.y) for lm in hand_landmarks]  # Normalize coordinates
# connections = mp.solutions.hands.HAND_CONNECTIONS  # Connections
# draw_landmarks_on_image(image_np, landmarks, connections)
#
# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()