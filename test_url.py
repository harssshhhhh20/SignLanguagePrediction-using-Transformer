import json

with open("WLASL_v0.3.json") as f:
    data = json.load(f)

youtube = 0
aslbrick = 0
other = 0

for sign in data[:20]:
    for instance in sign["instances"]:
        url = instance["url"]

        if "youtube" in url:
            youtube += 1
        elif "aslbrick" in url:
            aslbrick += 1
        else:
            other += 1

print("ASLBricks:", aslbrick)
print("YouTube:", youtube)
print("Other:", other)
print("Total:", aslbrick + youtube + other)