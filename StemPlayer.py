import pygame

STEMS = ["bass", "drums", "other", "vocals"]

class StemPlayer(object):
    def __init__(self, song_name: str):
        self.song_name = song_name
        self.stems = {}

        # Initialize the pygame mixer
        pygame.mixer.init()

        # Load stems
        for stem_name in STEMS:
            path = f"Stems/{song_name}/{stem_name}.wav"

            # Load the sound into pygame
            pygame_sound = pygame.mixer.Sound(path)

            self.stems[stem_name] = pygame_sound
            print(f"{stem_name} has been loaded")

            # Play the stem audio directly (non-blocking)
            pygame_sound.play()

    def set_stem_volume(self, stem_name: str, volume: float):
        self.stems[stem_name].set_volume(volume)