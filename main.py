import cv2

from HandTracker import HandTracker
from StemPlayer import StemPlayer

if __name__ == "__main__":
    hand_tracker = HandTracker(model_path='Models\\hand_landmarker.task')
    stem_player = StemPlayer("NEW-MAGIC-WAND")

    looped = False
    # hand_tracker.run()
    while hand_tracker.cap.isOpened():
        hand_tracker.update()

        #handle stem volumes
        stem_player.set_stem_volume("vocals", hand_tracker.left_index_finger_closure)
        stem_player.set_stem_volume("other", hand_tracker.left_middle_finger_closure)
        stem_player.set_stem_volume("drums", hand_tracker.right_index_finger_closure)
        stem_player.set_stem_volume("bass", hand_tracker.right_middle_finger_closure)

        if stem_player.is_looping and hand_tracker.distance_between_thumbs > 0.07:
            stem_player.stop_loop()
        elif not stem_player.is_looping and hand_tracker.distance_between_thumbs < 0.05:
            stem_player.start_loop()

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    stem_player.cleanup()
    hand_tracker.cap.release()
    cv2.destroyAllWindows()
