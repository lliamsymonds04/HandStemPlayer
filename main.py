import cv2

from HandTracker import HandTracker
from StemPlayer import StemPlayer

if __name__ == "__main__":
    hand_tracker = HandTracker(model_path='Models\\hand_landmarker.task')
    stem_player = StemPlayer("NEW-MAGIC-WAND")

    # hand_tracker.run()
    while hand_tracker.cap.isOpened():
        hand_tracker.update()

        #handle stem volumes
        stem_player.set_stem_volume("vocals", hand_tracker.left_index_finger_closure)
        stem_player.set_stem_volume("other", hand_tracker.left_middle_finger_closure)
        stem_player.set_stem_volume("drums", hand_tracker.right_index_finger_closure)
        stem_player.set_stem_volume("bass", hand_tracker.right_middle_finger_closure)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    hand_tracker.cap.release()
    cv2.destroyAllWindows()
