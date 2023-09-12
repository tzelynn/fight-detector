import matplotlib.pyplot as plt

def plot_loss_curves(train_loss, val_loss, epochs):
    plt.figure(figsize=(8, 4))
    plt.plot(train_loss, marker='s', color="salmon", linestyle="-")
    plt.plot(val_loss, marker='s', color="mediumseagreen", linestyle="-")
    plt.xticks(range(epochs), range(1, epochs + 1))
    plt.title("Training and Validation Loss Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])
    plt.grid(color="lightgrey")
    ymin = min(val_loss)
    xmin = val_loss.index(ymin)
    plt.annotate(
            f"min: {round(ymin,4)}", 
            xy = (xmin, ymin),
            xytext = (xmin-0.25, ymin+0.005),
            bbox = dict(boxstyle="round", fc="w", ec="k", lw=0.7),
            arrowprops = dict(arrowstyle="->")
        )
    plt.savefig(f"loss.jpg")

def plot_acc_curves(train_acc, val_acc, epochs):
    plt.figure(figsize=(8, 4))
    plt.plot(train_acc, marker='s', color="cornflowerblue", linestyle="-")
    plt.plot(val_acc, marker='s', color="mediumorchid", linestyle="-")
    plt.xticks(range(epochs), range(1, epochs + 1))
    plt.title("Training and Validation Accuracy Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Training", "Validation"])
    plt.grid(color="lightgrey")
    ymax = max(val_acc)
    xmax = val_acc.index(ymax)
    plt.annotate(
            f"max: {round(ymax,4)}", 
            xy = (xmax, ymax),
            xytext = (xmax-0.25, ymax-0.05),
            bbox = dict(boxstyle="round", fc="w", ec="k", lw=0.7),
            arrowprops = dict(arrowstyle="->")
        )
    plt.savefig(f"acc.jpg")