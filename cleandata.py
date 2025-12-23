import os
import shutil

BASE_DIR = "Extracted Faces"
MIN_IMAGES = 5

for folder in os.listdir(BASE_DIR):
    folder_path = os.path.join(BASE_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(images) < MIN_IMAGES:
        shutil.rmtree(folder_path)
