from HandTracker import HandTracker


if __name__ == "__main__":
    hand_tracker = HandTracker(model_path='Models\\hand_landmarker.task')
    hand_tracker.run()