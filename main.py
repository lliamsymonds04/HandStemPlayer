from HandTracker import HandTracker
from StemPlayer import StemPlayer

if __name__ == "__main__":
    hand_tracker = HandTracker(model_path='Models\\hand_landmarker.task')
    stem_player = StemPlayer()

    hand_tracker.run()

