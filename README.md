# Fight Detector

## Data source
This project uses data from https://github.com/seymanurakti/fight-detection-surv-dataset/. This cloned repository contains surveillance camera footage of 150 videos containing people fighting, and 150 videos without any people fighting. Each video is around 2 seconds long.

## Detection process
1. Each video is used as input into YOLOv8 (from `ultralytics`) to perform keypoint detection on each frame.
2. The set of keypoints across all frames of each video is used as input into an LSTM model. The output is a probability of the presence of people fighting in the video.
3. Using a fixed threshold value, the video is classified either as containing or not containing fighting.

## Training details
- The variables, parameters and hyperparameters used during the training process are fixed as follows:
    - Optimizer: Adam
    - Loss function: Binary Cross Entropy
    - Number of epochs: 20
    - Hidden dimension in LSTM: 512
    - Dropout: 0.07
    - Learning rate: 1e-5
    - Batch size: 2
- The data (total 300 videos) is split randomly as follows: 80% for training, 10% for validation, and 10% for testing
    - Keypoint detection is done altogether, then pickled for future rounds of training or testing
- The training and validation loss and accuracy across all epochs is plotted in [loss.jpg](visualisations/loss.jpg) and [acc.jpg](visualisations/acc.jpg).

## Results
The final trained LSTM model achieved a **66.7%** accuracy on the test data. The confusion matrix can be viewed [here](visualisations/cm_test.jpg). 
