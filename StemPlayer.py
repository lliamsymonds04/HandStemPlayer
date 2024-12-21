import csv
import json

import pygame
import time
import os
from pydub import AudioSegment

STEMS = ["bass", "drums", "other", "vocals"]

LOOPS_DIR = "LOOPS"

class StemPlayer(object):
    def __init__(self, song_name: str):
        self.song_name = song_name
        self.stems = {}
        self._new_stems = {}
        self.audio_segments = {}
        self.is_looping = False

        self._start_time = time.time()
        self._loop_start_time = 0
        self._loop_length = 1
        self._loop_offset = 0
        self._beat_duration = 0

        #load song data
        with open(f"Stems/{song_name}/data.json", "r") as file:
            data = json.load(file)

            self.bpm = data["bpm"]
            self.time_sig = data["time_sig"]

            beat_dur = 60/self.bpm
            self._loop_length = beat_dur * int(self.time_sig[0])

        # Initialize the pygame mixer
        pygame.mixer.init()

        # Load stems
        for stem_name in STEMS:
            path = f"Stems/{song_name}/{stem_name}.wav"

            # Load the sound into pygame
            stem = pygame.mixer.Sound(path)

            self.stems[stem_name] = stem
            self.audio_segments[stem_name] = AudioSegment.from_file(path)

            stem.play()

    def set_stem_volume(self, stem_name: str, volume: float):
        self.stems[stem_name].set_volume(volume)

    def start_loop(self):
        self.is_looping = True

        now = time.time()
        self._loop_start_time = now
        offset = now - self._start_time
        #create stems for loop and the stems to play after

        loop_start = int((offset - self._loop_length - self._loop_offset) * 1000) #convert to ms
        loop_end = int((offset - self._loop_offset) * 1000)

        for stem_name in STEMS:
            looped_audio = self.audio_segments[stem_name][loop_start:loop_end]
            post_loop_audio = self.audio_segments[stem_name][loop_start:]

            #export the loop
            loop_file_name = f"{LOOPS_DIR}/looped_{stem_name}.wav"
            looped_audio.export(loop_file_name, format="wav")

            post_loop_file_name = f"{LOOPS_DIR}/post_loop_{stem_name}.wav"
            post_loop_audio.export(post_loop_file_name, format="wav")

            self.stems[stem_name].stop()
            looped_stem = pygame.mixer.Sound(loop_file_name)

            looped_stem.play(loops=-1)
            self.stems[stem_name] = looped_stem

    def stop_loop(self):
        self.is_looping = False

        now = time.time()
        loop_time = now - self._loop_start_time
        loops = loop_time // self._loop_length

        self._loop_offset += (loops + 1) * self._loop_length
        splice_time = int((now - self._start_time - self._loop_offset) * 1000)


        for stem_name in STEMS:
            new_audio = self.audio_segments[stem_name][splice_time:]

            file_name = f"{LOOPS_DIR}/looped_{stem_name}.wav"
            new_audio.export(file_name, format="wav")

            self.stems[stem_name].stop()
            looped_stem = pygame.mixer.Sound(file_name)

            looped_stem.play()
            self.stems[stem_name] = looped_stem

    def cleanup(self):
        for item in os.listdir(LOOPS_DIR):
            item_path = os.path.join(LOOPS_DIR, item)  # Full path to the item

            # Check if it's a file
            if os.path.isfile(item_path):
                os.remove(item_path)  # Delete the file

#make the directory
try:
    os.mkdir(LOOPS_DIR)
except FileExistsError:
    # print("Loops directory already exists")
    pass
