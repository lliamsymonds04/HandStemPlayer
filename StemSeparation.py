import json
import subprocess
import os
from pathlib import Path
import shutil
import stat
import ctypes

def separate_music(file_path: str, song_name: str, bpm: str, time_sig: str, ):
    output_dir = f"Stems/{song_name.upper()}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        file_path.replace("\\", "/")
        os.makedirs(f"Stems/{song_name.upper()}", exist_ok=True)

        formatted_path = Path(file_path).as_posix()

        command = f'demucs --out "{output_dir}" {formatted_path}' #-n htdemucs_ft
        os.system(command)

        #move the stems out from the inside folder to the outer folder
        demucs_folder = os.path.join(output_dir,[item for item in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, item))][0]) #get the first sub folders
        stems_folder = os.path.join(demucs_folder, os.listdir(demucs_folder)[0])

        for stem in os.listdir(stems_folder):
            source = os.path.join(stems_folder, stem)
            shutil.move(source, output_dir)

        ctypes.windll.kernel32.DeleteFileW(demucs_folder)


        with open(output_dir + "/data.json", "w") as file:
            json.dump({"bpm": int(bpm), "time_sig": time_sig}, file)

    except subprocess.CalledProcessError as e:
        print(f"Error during Demucs execution: {e}")

    # try:
    #     if demucs_folder != "":
    #         os.chmod(demucs_folder, stat.S_IWRITE)
    #         os.remove(demucs_folder)
    # except PermissionError as e:
    #     print("failed to remove the temp folder")

# Input file
file_path = input("Enter the path to the audio file: ")
song_name = input("Enter the name of the song: ")
bpm = input("Enter the bpm: ")
time_sig = input("Enter the time signature (ie '4/4'): ")
separate_music(file_path, song_name, bpm, time_sig)