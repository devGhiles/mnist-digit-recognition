from abc import ABC, abstractmethod

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential
from keras.optimizers import RMSprop


class Model(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def evaluate(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save(self, filepath):
        pass

    @abstractmethod
    def load(self, filepath):
        pass


class CNNKerasModel(Model):

    def __init__(self, conv_dropout=0.25, dense_dropout=0.5):
        super().__init__()

        # set up a few hyperparameters
        self._conv_dropout = conv_dropout
        self._dense_dropout = dense_dropout

        # set up the model architecture
        self._build_model()

    def _build_model(self):
        self._model = Sequential()

        self._model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same',
            activation='relu', input_shape=(28, 28, 1)))
        self._model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
        self._model.add(MaxPool2D(pool_size=(2, 2)))
        self._model.add(Dropout(self._conv_dropout))

        self._model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self._model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self._model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        self._model.add(Dropout(self._conv_dropout))

        self._model.add(Flatten())
        self._model.add(Dense(256, activation='relu'))
        self._model.add(Dropout(self._dense_dropout))
        self._model.add(Dense(10, activation='softmax'))

        return None

    def train(self, X, y, X_valid=None, y_valid=None, epochs=10, batch_size=16, lr=0.001, verbose=0):
        # compile the model
        optimizer = RMSprop(lr=lr)
        self._model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # train the model
        kwargs = {
            'batch_size': batch_size,
            'epochs': epochs,
            'verbose': verbose
        }
        if X_valid is not None and y_valid is not None:
            kwargs['validation_data'] = (X_valid, y_valid)

        history = self._model.fit(X, y, **kwargs)

        return history.history

    def evaluate(self, X, y):
        loss, accuracy = self._model.evaluate(X, y)
        return {'loss': loss, 'accuracy': accuracy}

    def predict(self, X):
        return self._model.predict(X)

    def save(self, filepath):
        self._model.save_weights(filepath)

    def load(self, filepath):
        self._model.load_weights(filepath)
