# MNIST Digit Recognizer
[MNIST](http://yann.lecun.com/exdb/mnist/) is a toy dataset commonly used in Machine Learning and Computer Vision. It consists of black and white images of handwritten digits. The objective of this project is to create a Machine Learning model which identifies the digit contained in an image, and to make a submission to the [Kaggle Digit Recognizer competition](https://www.kaggle.com/c/digit-recognizer). The dataset used in this project is the one provided by the Kaggle competition; using another version of the dataset might result in using the Kaggle test data as training data and getting innacuratly high accuracy scores on the compitition leaderboard (this is why there are a lot of 100% accuracy scores on the leaderboard).

## Dependencies
 - Python 3.5.2 or higher (not tested on older versions)
 - The file **requirements.txt** contains the required Python libraries. They can be installed via pip using the command `$ pip install -r requirements.txt`

## Explore the data
Data exploration and analysis is a crutial step for solving Machine Learning problems. The jupyter notebook **eda.ipynb** contains a basic exploratory data analysis of the dataset.

## Train a Machine Learning model
The script **train.py** allows to train a model. Two models are implemented: a feedforward neural network and a convolutional neural network. The two models are defined in **core/model.py** by the classes `FeedForwardNeuralNetworkModel` and `CNNKerasModel` respectively.

Here is the command for training a CNN model:

    $ python train.py cnn data/train.csv data/valid.csv cnn.model cnn.report cnn_params.json

where *"cnn"* is the type of the model (can be *"cnn"* or *"ff"*), *data/train.csv* is the path of the training data, *data/valid.csv* is the path of the validation data, *cnn.model* is the path where the trained model is saved, *cnn.report* is the path where the report of the training is saved, and *cnn_params.json* defines the hyperparameters of the CNN. An example of *cnn_params.json* is included in this repository.
The command `$ python train.py -h` provides more information about the **train.py** script.

## Make predictions and generate the submission file
The script **predict.py** allows to compute the predicted digit values of a given test file and generate a submission file suitable for the [Kaggle Digit Recognizer competition](https://www.kaggle.com/c/digit-recognizer). The full command is:
```
python predict.py cnn cnn.model data/test.csv submission.csv
```
where *"cnn"* is the type of the model (can be *"cnn"* or *"ff"*), *cnn.model* is the path of the trained model, *data/test.csv* is the path of the test data, and *submission.csv* is the path of the generated submission file, which is going to be uploaded to Kaggle.
The command `$ python predict.py -h` provides more information about the **predict.py** script.