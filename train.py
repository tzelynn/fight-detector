from data import get_datafiles, get_keypoints, get_dataloader
from plots import plot_acc_curves, plot_loss_curves
from lstm import ActionRecognitionLSTM

import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
from torch.optim import Adam
from ultralytics import YOLO


def trainLSTM(batch_size, max_frames=142, max_ppl=13, epochs=20, dirs=None):

    if dirs:
        fightDir, noFightDir = dirs
        X_train, X_val, X_test, y_train, y_val, y_test = get_datafiles(fightDir, noFightDir)
        file = open("y_train", "ab")
        pickle.dump(y_train, file)
        file.close()
        file = open("y_val", "ab")
        pickle.dump(y_val, file)
        file.close()
        file = open("y_test", "ab")
        pickle.dump(y_test, file)
        file.close()

        print("detecting from scratch...")
        detectionModel = YOLO('yolov8n-pose.pt')
        padding = np.array([[[0, 0]] * 17] * max_ppl, dtype=np.float32)
        X_train_kp = get_keypoints(X_train, max_frames, max_ppl, detectionModel, padding)
        X_val_kp = get_keypoints(X_val, max_frames, max_ppl, detectionModel, padding)
        X_test_kp = get_keypoints(X_test, max_frames, max_ppl, detectionModel, padding)
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
    input_dim = max_ppl * 34  # * max frames
    output_dim = 1
    hidden_dim = 1024
    dropout = 0.07
    lr = 1e-4
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

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

        if avg_val_loss == min(epoch_val_loss):
            best_epoch_sd = recognitionModel.state_dict()
            best_epoch = epoch
            best_val_loss = avg_val_loss
            best_val_acc = avg_val_acc
            best_cm = cm
    
    torch.save(best_epoch_sd, "ARLSTM.pth")
    print(f"Saved model from epoch {best_epoch}.")
    print(f"lowest val loss {best_val_loss}, val acc {best_val_acc}")
    disp = ConfusionMatrixDisplay(best_cm)
    disp.plot()
    disp.figure_.savefig("cm.jpg")

    file_stats = open("stats", "ab")
    stats = {"train_loss": epoch_train_loss,
             "train_acc": epoch_train_acc,
             "val_loss": epoch_val_loss,
             "val_acc": epoch_val_acc
             }
    pickle.dump(stats, file_stats)
    file_stats.close()

    plot_loss_curves(epoch_train_loss, epoch_val_loss, epochs)
    plot_acc_curves(epoch_train_acc, epoch_val_acc, epochs)

    return epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc


def acc_fn(preds, labels):
    preds = preds.round()
    corr = (preds == labels).float()
    return corr.sum() / len(corr)

def train_loop(model, train_loader, device, loss_fn, optimizer):
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


def eval_loop(model, val_loader, device, loss_fn):

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
    fightDir = "fight-detection-surv-dataset/fight"
    noFightDir = "fight-detection-surv-dataset/noFight"
    trainLSTM(batch_size)  # , dirs=[fightDir, noFightDir])