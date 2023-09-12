import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_datafiles(fightDir, noFightDir):
    fightFiles = format_files(fightDir)
    noFightFiles = format_files(noFightDir)

    X = fightFiles + noFightFiles
    y = [1] * len(fightFiles) + [0] * len(noFightFiles)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=0.5, random_state=42, shuffle=True)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def format_files(dir):
    files = os.listdir(dir)
    files = [os.path.join(dir, f) for f in files]
    return files

def detect(input, model):
    results = model.predict(
        source=input, save=True, stream_buffer=True, stream=True)
    return results

def get_keypoints(X, max_frames, max_ppl, detectionModel, padding):
    """X_kp = [[vid1_kp: [frame1_kp: [x, y], [x, y]]], [vid2_kp]] 
    dim = num_vids, max_frames, 17, 2
    """
    X_kp = []
    people_pad = np.array([[[0,0]] * 17])
    for file in X:
        results = detect(file, detectionModel)
        frames = 1
        frame_kp = []
        for result in results:
            kp = result.keypoints.xyn.cpu().numpy()
            num_ppl = kp.shape[0]
            if kp.shape[1] == 0:
                kp = np.concatenate([people_pad] * max_ppl)
            elif num_ppl < max_ppl:
                num_ppl_pad = max_ppl - num_ppl
                kp = np.concatenate([kp, np.concatenate([people_pad] * num_ppl_pad)])
            frame_kp.append(kp)
            frames += 1
            if frames > max_frames:
                break
        for _ in range(frames, max_frames+1):
            frame_kp.append(padding)
        X_kp.append(frame_kp)

    return X_kp

def get_dataloader(X, y, batch_size):
    X = np.array(X)
    y = np.array(y)
    data = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return loader
