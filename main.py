import cv2
import os

from rapidfuzz import process, fuzz

from HandTracker import HandTracker
from StemPlayer import StemPlayer


def get_file_names(directory):
    return [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

# Example usage
STEMS_FOLDER = f"{os.getcwd()}\\Stems"
Song_names = get_file_names(STEMS_FOLDER)
print("Available Songs:")
print(Song_names)

if __name__ == "__main__":

    song_name = ""

    while song_name == "":
        user_song = input("Enter song name: ").upper()

        result = process.extractOne(user_song, Song_names)

        if result and result[1] > 70:
            song_name = result[0]
            print(f"Selected {song_name}")
        else:
            print("Couldn't find that song, try again.")

    hand_tracker = HandTracker(model_path='Models\\hand_landmarker.task', camera_size_factor=0.6)
    stem_player = StemPlayer(song_name)

    while hand_tracker.cap.isOpened():
        hand_tracker.update()

        #handle stem volumes
        stem_player.set_stem_volume("vocals", hand_tracker.left_index_finger_closure)
        stem_player.set_stem_volume("other", hand_tracker.left_middle_finger_closure)
        stem_player.set_stem_volume("drums", hand_tracker.right_index_finger_closure)
        stem_player.set_stem_volume("bass", hand_tracker.right_middle_finger_closure)

        if stem_player.is_looping and hand_tracker.distance_between_thumbs > 0.1:
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
