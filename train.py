"""Script to train the LSTM from video inputs"""
from data import get_datafiles, get_keypoints, get_dataloader
from plots import plot_acc_curves, plot_loss_curves
from lstm import ActionRecognitionLSTM

import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Callable, List, Optional, Tuple
from ultralytics import YOLO


def trainLSTM(
        batch_size: int,
        max_frames:int =142,
        max_ppl: int =13,
        epochs: int =20,
        dirs: Optional[List[str]] =None
    ) -> Tuple[List[float]]:
    """Obtain keypoint detections, prepare LSTM input, and run training and
    evaluation for LSTM. Plot and save learning curves, and save confusion
    matrix from validation of best model.
    
    Inputs:
        batch_size (int): batch size for LSTM training
        max_frames (int): max number of video frames to use as input
        max_ppl (int): max number of people detected in each frame
        epochs (int): number of training epochs for LSTM
        dirs (Optional[List[str]]): list of directories, 
            [fight_dir, no_fight_dir], if keypoints not detected before
    Output:
        Tuple containing 4 lists of floats
            1. epoch_train_loss: list of train loss per epoch
            2. epoch_train_acc: list of train accuracy per epoch
            3. epoch_val_loss: list of validation loss per epoch
            4. epoch_val_acc: list of validation accuracy per epoch
    """

    if dirs:
        # need to perform keypoint detection using YOLOv8 first
        fight_dir, no_fight_dir = dirs
        X_train, X_val, X_test, y_train, y_val, y_test = get_datafiles(
            fight_dir, no_fight_dir)

        # pickle the video labels and dir
        file = open("y_train", "ab")
        pickle.dump(y_train, file)
        file.close()
        file = open("y_val", "ab")
        pickle.dump(y_val, file)
        file.close()
        file = open("y_test", "ab")
        pickle.dump(y_test, file)
        file.close()
        file = open("X_train_dir", "ab")
        pickle.dump(X_train, file)
        file.close()
        file = open("X_val_dir", "ab")
        pickle.dump(X_val, file)
        file.close()
        file = open("X_test_dir", "ab")
        pickle.dump(X_test, file)
        file.close()

        print("detecting keypoints from scratch...")
        detectionModel = YOLO('yolov8n-pose.pt')
        padding = np.array(
            [[[0, 0]] * 17] * max_ppl, dtype=np.float32)  # frame padding
        X_train_kp = get_keypoints(
            X_train, max_frames, max_ppl, detectionModel, padding)
        X_val_kp = get_keypoints(
            X_val, max_frames, max_ppl, detectionModel, padding)
        X_test_kp = get_keypoints(
            X_test, max_frames, max_ppl, detectionModel, padding)

        # pickle the detected keypoints
        file = open("X_train", "ab")
        pickle.dump(X_train_kp, file)
        file.close()
        file = open("X_val", "ab")
        pickle.dump(X_val_kp, file)
        file.close()
        file = open("X_test", "ab")
        pickle.dump(X_test_kp, file)
        file.close()
    else:
        # use previously detected keypoints
        print("using pickled detections...")
        file_p = open("X_train", "rb")
        X_train_kp = pickle.load(file_p)
        file_p.close()
        file_p = open("X_val", "rb")
        X_val_kp = pickle.load(file_p)
        file_p.close()

        file_p = open("y_train", "rb")
        y_train = pickle.load(file_p)
        file_p.close()
        file_p = open("y_val", "rb")
        y_val = pickle.load(file_p)
        file_p.close()

    train_loader = get_dataloader(X_train_kp, y_train, batch_size)
    val_loader = get_dataloader(X_val_kp, y_val, batch_size)

    # hyperparams
    input_dim = max_ppl * 34
    output_dim = 1
    hidden_dim = 512
    dropout = 0.07
    lr = 1e-5

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    # initialise LSTM model, loss function and optimizer
    recognitionModel = ActionRecognitionLSTM(
        input_dim, output_dim, hidden_dim, dropout)
    recognitionModel.to(device)
    loss_fn = nn.BCELoss()
    optimizer = Adam(recognitionModel.parameters(), lr = lr)

    epoch_train_loss = []
    epoch_train_acc = []
    epoch_val_loss = []
    epoch_val_acc = []
    best_epoch_sd = {}
    best_epoch = 0
    best_val_loss = 0
    best_val_acc = 0
    best_cm = None

    # train and evaluate LSTM model
    for epoch in range(1, epochs+1):
        print(f">>> EPOCH {epoch} / {epochs}")

        train_losses, train_acc = train_loop(
            recognitionModel, train_loader, device, loss_fn, optimizer)
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_acc = sum(train_acc) / len(train_acc)
        epoch_train_loss.append(avg_train_loss)
        epoch_train_acc.append(avg_train_acc)

        avg_val_loss, avg_val_acc, preds_lst, cm = eval_loop(
            recognitionModel, val_loader, device, loss_fn)
        epoch_val_loss.append(avg_val_loss)
        epoch_val_acc.append(avg_val_acc)

        print(">>> EPOCH STATS")
        print("Average train loss:", avg_train_loss)
        print("Average train acc:", avg_train_acc)
        print("Average val loss:", avg_val_loss)
        print("Average val acc:", avg_val_acc)
        print("preds_lst", preds_lst)
        print()

        # record best model based on average validation loss
        if avg_val_loss == min(epoch_val_loss):
            best_epoch_sd = recognitionModel.state_dict()
            best_epoch = epoch
            best_val_loss = avg_val_loss
            best_val_acc = avg_val_acc
            best_cm = cm

    # save the best model and plot confusion matrix
    torch.save(best_epoch_sd, "ARLSTM.pth")
    print(f"Saved model from epoch {best_epoch}.")
    print(f"lowest val loss {best_val_loss}, val acc {best_val_acc}")
    disp = ConfusionMatrixDisplay(best_cm)
    disp.plot()
    disp.figure_.savefig("cm.jpg")

    # save training and validation stats
    file_stats = open("stats", "ab")
    stats = {"train_loss": epoch_train_loss,
             "train_acc": epoch_train_acc,
             "val_loss": epoch_val_loss,
             "val_acc": epoch_val_acc
             }
    pickle.dump(stats, file_stats)
    file_stats.close()

    # plot learning curves
    plot_loss_curves(epoch_train_loss, epoch_val_loss, epochs)
    plot_acc_curves(epoch_train_acc, epoch_val_acc, epochs)

    return epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc


def acc_fn(
        preds: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
    """Calculate the accuracy of the predictions.
    Inputs:
        preds (torch.Tensor): predictions from the model
        labels (torch.Tensor): ground truth labels
    Output:
        Tensor containing accuracy value
    """
    preds = preds.round()
    corr = (preds == labels).float()
    return corr.sum() / len(corr)

def train_loop(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        device: torch.device,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: torch.optim
    ) -> Tuple[List[float]]:
    """Train loop for each epoch.
    Inputs:
        model (nn.Module): LSTM model
        train_loader (torch DataLoader): DataLoader for train input
        device (torch.device): device to conduct training
        loss_fn (Callable): takes in prediction and ground truth tensors
            and returns loss (as tensor)
        optimizer (torch.optim): optimizer for training
    Output:
        Tuple containing 2 lists:
            1. train_losses: contains training losses from each batch
            2. train_acc: contains training accuracy from each batch
    """
    train_losses = []
    train_acc = []

    model.train()

    batch_count = 1

    for inputs, labels in train_loader:

        if batch_count == 1 or batch_count % 50 == 0:
            print(f"Training Batch {batch_count} / {len(train_loader)}") 

        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()

        # model training
        outputs = model(inputs)  # dim: batch_size, max_frames, x
        preds = outputs.mean(dim=1).flatten()
        loss = loss_fn(preds, labels.float())
        loss.backward()
        optimizer.step()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # recording training stats
        train_losses.append(loss.item())
        train_acc.append(acc_fn(preds, labels).item())

        # updates for next iteration
        batch_count += 1

    # returns lists of training loss and acc for this epoch
    return train_losses, train_acc


def eval_loop(
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ):
    """Evaluation loop for each epoch.
    Inputs:
        model (nn.Module): LSTM model
        val_loader (torch DataLoader): DataLoader for validation input
        device (torch.device): device to conduct training
        loss_fn (Callable): takes in prediction and ground truth tensors
            and returns loss (as tensor)
    Output:
        Tuple containing 2 lists:
            1. avg_val_loss: contains training losses from each batch
            2. train_acc: contains training accuracy from each batch
    """

    val_losses = 0
    val_acc = 0
    preds_lst = []
    gt_lst = []

    model.eval()
    for inputs, labels in val_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = outputs.mean(dim=1).flatten()
        loss = loss_fn(preds, labels.float())

        val_losses += loss.item()
        val_acc += acc_fn(preds, labels).item()
        preds_lst.extend(preds.tolist())
        gt_lst.extend(labels.tolist())
    
    cm = confusion_matrix(gt_lst, list(map(round, preds_lst)))

    # returns average val loss and acc
    return val_losses / len(val_loader), val_acc / len(val_loader), preds_lst, cm


if __name__ == "__main__":
    batch_size = 2
    fight_dir = "fight-detection-surv-dataset/fight"
    no_fight_dir = "fight-detection-surv-dataset/noFight"
    trainLSTM(batch_size)  #, dirs=[fight_dir, no_fight_dir])