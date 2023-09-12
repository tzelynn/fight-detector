import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple, Union
import ultralytics


def get_datafiles(
        fight_dir: str,
        no_fight_dir: str
    ) -> Tuple[List[Union[str, int]]]:
    """Gets required data files for keypoint detection and training.
    Inputs:
        fight_dir (str): path to folder containing fighting videos
        no_fight_dir (str): path to folder containing non-fighting videos
    Output:
        Tuple containing 6 lists:
            1. X_train: str paths of training videos
            2. X_val: str paths of validation videos
            3. X_test: str paths of test videos
            4. y_train: int labels for training videos
            5. y_val: int labels for validation videos
            6. y_test: int labels for test videos
    """
    fightFiles = format_files(fight_dir)
    noFightFiles = format_files(no_fight_dir)

    X = fightFiles + noFightFiles
    y = [1] * len(fightFiles) + [0] * len(noFightFiles)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=0.5, random_state=42, shuffle=True)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def format_files(dir: str) -> List[str]:
    """Get full path names to all videos
    Inputs:
        dir (str): path to video folder
    Output:
        files (List[str]): list of paths to all videos in dir
    """
    files = os.listdir(dir)
    files = [os.path.join(dir, f) for f in files]
    return files

def detect(
        input: str,
        model: ultralytics.engine.model.Model
    ) -> ultralytics.engine.results.Results:
    """Detects keypoints in a video
    Inputs:
        input (str): path to video to feed into model
        model: chosen YOLO model to do keypoint detection
    Output:
        results: detection output of YOLO model
    """
    results = model.predict(
        source=input, save=True, stream_buffer=True, stream=True)
    return results

def get_keypoints(
        X: List[str],
        max_frames: int,
        max_ppl: int,
        detection_model: ultralytics.engine.model.Model,
        padding: np.ndarray
    ) -> List[List[np.ndarray]]:
    """Run keypoint detection for dataset and pad detections.
    Inputs:
        X (List[str]): list of paths to videos
        max_frames (int): max number of frames per video
        max_ppl (int): max number of people per frame
        detection_model: YOLO model used for keypoint detection
        padding: array of 0s to make total frames per vid consistent
    Output:
        X_kp: each video is broken down into a list of detections per frame (
            dim = num_vids, max_frames, 17, 2)
    """
    X_kp = []
    people_pad = np.array([[[0,0]] * 17])
    for file in X:
        results = detect(file, detection_model)
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

def get_dataloader(
        X: List[List[np.ndarray]],
        y: List[int],
        batch_size: int
    ) -> torch.utils.data.DataLoader:
    """Get torch dataloader for model training/evaluation.
    Inputs:
        X: keypoints of each frame of each video
        y: labels of each video
        batch_size (int): batch size for training/evaluation
    Output:
        loader: torch dataloader
    """
    X = np.array(X)
    y = np.array(y)
    data = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return loader
