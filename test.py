from data import get_dataloader
from lstm import ActionRecognitionLSTM
from train import eval_loop

from sklearn.metrics import ConfusionMatrixDisplay
import pickle
import torch


def load_model(path):
    model = ActionRecognitionLSTM(
        input_dim=442, output_dim=1, hidden_dim=512)
    model.load_state_dict(torch.load(path))
    return model

def test(model, X_file, y_file):

    file = open(X_file, "rb")
    X_test = pickle.load(file)
    file.close()
    file = open(y_file, "rb")
    y_test = pickle.load(file)
    file.close()

    test_loader = get_dataloader(X_test, y_test, batch_size=2)
    device = torch.device("cpu")
    loss_fn = torch.nn.BCELoss()

    test_loss, test_acc, _, cm = eval_loop(
        model, test_loader, device, loss_fn)
    
    print("Loss:", test_loss)
    print("Acc:", test_acc)

    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    disp.figure_.savefig("cm_test.jpg")

if __name__ == "__main__":
    model = load_model("ARLSTM.pth")
    test(model, X_file="X_test", y_file="y_test")