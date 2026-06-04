import json
import os
from collections import Counter

VIDEO_DIR = "videos"

with open("WLASL_v0.3.json", "r") as f:
    data = json.load(f)

available_classes = Counter()

for sign in data:

    gloss = sign["gloss"]

    for instance in sign["instances"]:

        video_id = instance["video_id"]

        video_path = os.path.join(
            VIDEO_DIR,
            f"{video_id}.mp4"
        )

        if os.path.exists(video_path):
            available_classes[gloss] += 1

print("\nTop 30 Classes By Available Videos\n")

for gloss, count in available_classes.most_common(30):
    print(f"{gloss}: {count}")

print("\nStatistics")
print("Unique Classes:", len(available_classes))
print(
    "Total Available Videos:",
    sum(available_classes.values())
)