"""Script to test the LSTM using pickled inputs"""
from data import get_dataloader
from lstm import ActionRecognitionLSTM
from train import eval_loop

from sklearn.metrics import ConfusionMatrixDisplay
import pickle
import torch


def load_model(path: str) -> torch.nn.Module:
    """Load model using weights saved during training
    Inputs:
        path (str): path to model weights
    Output:
        LSTM model with loaded weights
    """
    model = ActionRecognitionLSTM(
        input_dim=442, output_dim=1, hidden_dim=512)
    model.load_state_dict(torch.load(path))
    return model

def test(
        model: torch.nn.Module,
        X_file: str,
        y_file: str
    ):
    """Test model performance on unseen data
    Inputs:
        model (torch Module): test model
        X_file (str): path to pickled file of keypoint detections
        y_file (str): path to pickled file of labels
    Output:
        None
    """
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
    disp.figure_.savefig("visualisations/cm_test.jpg")

if __name__ == "__main__":
    model = load_model("ARLSTM.pth")
    test(model, X_file="pickled/X_test", y_file="pickled/y_test")