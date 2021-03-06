from abc import ABC, abstractmethod

from keras.utils.np_utils import to_categorical


class Encoder(ABC):
    """Base class for all the data encoders.

    An encoder object transforms the original data which is a
    pandas DataFrame into a format suitable for use by the
    underlying model.
    """

    def __init__(self):
        super(Encoder, self).__init__()

    @abstractmethod
    def encode_X(self, df):
        pass

    @abstractmethod
    def encode_y(self, df):
        pass

    def encode_X_and_y(self, df):
        X = self.encode_X(df)
        y = self.encode_y(df)
        return X, y


class CNNKerasEncoder(Encoder):
    """Encoder class for CNNKerasModel."""

    def __init__(self):
        super(CNNKerasEncoder, self).__init__()

    def encode_X(self, df):
        # Remove the label from the data frame
        try:
            df_without_label = df.drop(labels=['label'], axis=1)
        except ValueError:
            df_without_label = df

        # Transform the data frame into a numpy array
        X = df_without_label.values
        
        # normalize the values between 0.0 and 1.0
        X = X / 255.0

        # reshape the image in 3D for keras input (28x28, 1 canal)
        X = X.reshape(-1, 28, 28, 1)
        
        return X

    def encode_y(self, df):
        # Get the labels
        y = df['label'].values

        # One-hot encoding
        return to_categorical(y, num_classes=10)


class FeedForwardNeuralNetworkEncoder(Encoder):
    """Encoder class for FeedForwardNeuralNetworkModel."""

    def __init__(self):
        super(FeedForwardNeuralNetworkEncoder, self).__init__()

    def encode_X(self, df):
        # Remove the label from the data frame
        try:
            df_without_label = df.drop(labels=['label'], axis=1)
        except ValueError:
            df_without_label = df

        # Transform the data frame into a numpy array
        X = df_without_label.values
        
        # normalize the values between 0.0 and 1.0
        X = X / 255.0
        
        return X

    def encode_y(self, df):
        # Get the labels
        y = df['label'].values

        # One-hot encoding
        return to_categorical(y, num_classes=10)
