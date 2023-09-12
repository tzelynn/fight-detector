from detect import detect

import cv2
import os
from ultralytics import YOLO

fightDir = "fight-detection-surv-dataset/fight"
noFightDir = "fight-detection-surv-dataset/noFight"
fightFiles = os.listdir(fightDir) 
noFightFiles = os.listdir(noFightDir)
fightFiles = [os.path.join(fightDir, f) for f in fightFiles]
noFightFiles = [os.path.join(noFightDir, f) for f in noFightFiles]
files = fightFiles + noFightFiles

max_frames = 0
min_frames = 100
for file in files:
    cap = cv2.VideoCapture(file)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = max(max_frames, num_frames)
    min_frames = min(min_frames, num_frames)

print(min_frames)  # 20
print(max_frames)  # 142


max_ppl = 0
min_ppl = 100
model = YOLO('yolov8n-pose.pt')
for file in files:
    results = detect(file, model)
    for result in results:
        kp = result.keypoints.xyn.cpu().numpy()
        num_ppl = kp.shape[0]
        max_ppl = max(max_ppl, num_ppl)
        min_ppl = min(min_ppl, num_ppl)

print(max_ppl)  # 13
print(min_ppl)  # 1