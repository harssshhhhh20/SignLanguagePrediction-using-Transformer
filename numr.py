import json
import os

with open("WLASL_v0.3.json") as f:
    data = json.load(f)

selected = {item["gloss"] for item in data[:20]}

available = 0
missing = 0

for sign in data:
    if sign["gloss"] not in selected:
        continue

    for instance in sign["instances"]:
        video_id = instance["video_id"]

        if os.path.exists(f"videos/{video_id}.mp4"):
            available += 1
        else:
            missing += 1

print("Available:", available)
print("Missing:", missing)