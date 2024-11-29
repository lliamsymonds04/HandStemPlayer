import cv2
import mediapipe as mp

from PointUtil import get_distance, convert_to_pixels, value_to_color, get_mid_point, lerp

# Initialize Mediapipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("CUDA is available!")
else:
    print("CUDA is not available.")

base_options = mp.tasks.BaseOptions(model_asset_path='gesture_recognizer.task',
delegate=mp.tasks.BaseOptions.Delegate.GPU)

# Open webcam
cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Hand Tracking", 800, 600)

cap = cv2.VideoCapture(0)

MIN_CIRCLE_SIZE = 2
MAX_CIRCLE_SIZE = 15

MID_POINT_MAX_LEN = 0.35
PINCH_MAX_LEN = 0.25

# connection_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3)  # Green lines
# landmark_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)   # Red dots


states = ["Calibrating"]
state = 0

with mp_hands.Hands(
        static_image_mode=False,  # Real-time detection
        max_num_hands=2,  # Detect up to 2 hands
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # # Flip the image horizontally for a mirrored effect
        # frame = cv2.flip(frame, 1)

        # Convert the image to RGB (Mediapipe uses RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = hands.process(rgb_frame)

        height, width, depth = frame.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections on the original frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, )

            if state == 0:
                pass

            # if len(results.multi_hand_landmarks) == 2:
            #     index_0 = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            #     thumb_0 = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP]
            #     index_1 = results.multi_hand_landmarks[1].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            #     thumb_1 = results.multi_hand_landmarks[1].landmark[mp_hands.HandLandmark.THUMB_TIP]

            #     index_0_p = convert_to_pixels(index_0, frame)
            #     thumb_0_p = convert_to_pixels(thumb_0, frame)
            #     index_1_p = convert_to_pixels(index_1, frame)
            #     thumb_1_p = convert_to_pixels(thumb_1, frame)

            #     mid_point_0 = get_mid_point(index_0_p, thumb_0_p)
            #     mid_point_1 = get_mid_point(index_1_p, thumb_1_p)

            #     print(index_0.z)

            #     # dist = get_distance(index_0, index_1)

            #     # text_pos = (int(index_0.x * width), int(index_0.y * height))
            #     # cv2.putText(
            #     #     frame,
            #     #     str(dist),
            #     #     text_pos,
            #     #     cv2.FONT_HERSHEY_SIMPLEX,  # Font type
            #     #     1,                         # Font scale
            #     #     (0, 255, 0),               # Text color (green)
            #     #     2,                         # Thickness
            #     #     cv2.LINE_AA 

            #     # )

            #     #middle line
            #     mid_point_dist = get_distance(mid_point_0, mid_point_1)

            #     mid_line_float = min(mid_point_dist/(width * MID_POINT_MAX_LEN), 1)
            #     mid_point_colour = value_to_color(mid_line_float, "cool")
            #     mid_circle_size = int(lerp(MIN_CIRCLE_SIZE, MAX_CIRCLE_SIZE, mid_line_float))

            #     cv2.circle(frame, mid_point_0, mid_circle_size, mid_point_colour, cv2.FILLED)
            #     cv2.circle(frame, mid_point_1, mid_circle_size, mid_point_colour, cv2.FILLED)
            #     cv2.line(frame, mid_point_0, mid_point_1, mid_point_colour, 3)

            #     #left line
            #     left_point_dist = get_distance(index_0_p, thumb_0_p)

            #     left_line_float = min(left_point_dist/(width * PINCH_MAX_LEN), 1)
            #     left_point_colour = value_to_color(left_line_float, "hot")
            #     left_circle_size = int(lerp(MIN_CIRCLE_SIZE, MAX_CIRCLE_SIZE, left_line_float))

            #     cv2.circle(frame, index_0_p, left_circle_size, left_point_colour, cv2.FILLED)
            #     cv2.circle(frame, thumb_0_p, left_circle_size, left_point_colour, cv2.FILLED)
            #     cv2.line(frame, index_0_p, thumb_0_p, left_point_colour, 3)

            #     #right line
            #     right_point_dist = get_distance(index_1_p, thumb_1_p)

            #     right_line_float = min(right_point_dist/(width * PINCH_MAX_LEN), 1)
            #     right_point_colour = value_to_color(right_line_float, "spring")
            #     right_circle_size = int(lerp(MIN_CIRCLE_SIZE, MAX_CIRCLE_SIZE, right_line_float))

            #     cv2.circle(frame, index_1_p, right_circle_size, right_point_colour, cv2.FILLED)
            #     cv2.circle(frame, thumb_1_p, right_circle_size, right_point_colour, cv2.FILLED)
            #     cv2.line(frame, index_1_p, thumb_1_p, right_point_colour, 3)

        # Display the frame
        cv2.imshow('Hand Tracking', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()