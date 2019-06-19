from abc import ABC, abstractmethod

from keras.utils.np_utils import to_categorical


class Encoder(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def encode_X(self, df):
        pass

    @abstractmethod
    def encode_y(self, df):
        pass

    def encode_X_and_y(self, df):
        X = encode_X(df)
        y = encode_y(df)
        return X, y


class CNNKerasEncoder(Encoder):

    def __init__(self):
        super().__init__()

    def encode_X(self, df):
        try:
            df_without_label = df.drop(labels=['label'], axis=1)
        except ValueError:
            df_without_label = df

        X = df_without_label.values
        
        # normalize the values between 0.0 and 1.0
        X = X / 255.0

        # reshape the image in 3D for keras input (28x28, 1 canal)
        X = X.reshape(-1, 28, 28, 1)
        
        return X

    def encode_y(self, df):
        y = df['label'].values
        return to_categorical(y, num_classes=10)
