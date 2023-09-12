from ultralytics import YOLO
import cv2

# constants
HEIGHT = 640
WIDTH = 480
IMAGE = "fight-detection-surv-dataset/fight/fi001.mp4"
# IMAGE = "bus.jpg"
IMAGE2 = "fight-detection-surv-dataset/noFight/nofi001.mp4"

# load model
model = YOLO('yolov8n-pose.pt')

def detect(input, model):
    results = model.predict(
        source=input, save=True, stream_buffer=True, stream=True)
    return results

# results = detect(IMAGE, model)

# frames = 0
# kps = []
# for result in results:
#     frames += 1
#     result_keypoint = result.keypoints.xyn.cpu().numpy()
#     kps.append(result_keypoint)
    # print(len(result_keypoint))  # 5
    # print(len(result_keypoint[0]))  # 7

# print(kps)

# import pickle

# file = open("pickle", "ab")
# pickle.dump(kps, file)
# file.close()

# file_p = open("pickle", "rb")
# kps_p = pickle.load(file_p)
# file_p.close()
# print(kps_p)

# for i in range(1, len(kps)):
#     prev = kps[i-1]
#     curr = kps[i]
#     print(curr-prev)

# print(frames)

# results = detect(IMAGE2, model)

# frames = 0
# for result in results:
#     frames += 1
#     result_keypoint = result.keypoints.xyn.cpu().numpy()
#     # print(len(result_keypoint))  # 5
#     # print(len(result_keypoint[0]))  # 7

# print(frames)
# print(result)
