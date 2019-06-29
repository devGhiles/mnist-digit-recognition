from abc import ABC, abstractmethod
import gzip
import json

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential
from keras.optimizers import RMSprop
import numpy as np

from .report import CNNKerasTrainingReport


class Model(ABC):

    @classmethod
    def load(cls, filepath):
        return cls.load(filepath)

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


class CNNKerasModel(Model):

    @staticmethod
    def _transform_weights_to_lists(weights):
        lists = []
        for a in weights:
            lists.append(a.tolist())
        
        return lists

    @staticmethod
    def _transform_lists_to_weights(lists):
        weights = []
        for l in lists:
            weights.append(np.array(l))
        
        return weights

    @staticmethod
    def load(filepath):
        # load the weights and the hyperparameters
        with gzip.open(filepath, 'rb') as f:
            json_obj = json.loads(f.read().decode('utf-8'))

        # set the weights and the hyperparameters
        model = CNNKerasModel()
        model._conv_dropout = json_obj['conv_dropout']
        model._dense_dropout = json_obj['dense_dropout']
        model._model.set_weights(CNNKerasModel._transform_lists_to_weights(json_obj['weights']))

        # compile the model
        model._compile_model()

        return model

    def __init__(self, conv_dropout=0.25, dense_dropout=0.5):
        super().__init__()

        # set up a few hyperparameters
        self._conv_dropout = conv_dropout
        self._dense_dropout = dense_dropout

        # used for compiling the keras model
        self._optimizer = None

        # set up the model architecture
        self._build_model()

    def _build_model(self):
        self._model = Sequential()

        self._model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same',
            activation='relu', input_shape=(28, 28, 1)))
        self._model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
        self._model.add(MaxPool2D(pool_size=(2, 2)))
        if self._conv_dropout != 0.0:
            self._model.add(Dropout(self._conv_dropout))

        self._model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self._model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self._model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        if self._conv_dropout != 0.0:
            self._model.add(Dropout(self._conv_dropout))

        self._model.add(Flatten())
        self._model.add(Dense(256, activation='relu'))
        if self._dense_dropout != 0.0:
            self._model.add(Dropout(self._dense_dropout))
        self._model.add(Dense(10, activation='softmax'))

        return None

    def _compile_model(self, optimizer=None):
        # define the optimizer
        if optimizer:
            self._optimizer = optimizer
        elif not self._optimizer:
            self._optimizer = RMSprop(lr=0.001)

        # compile the model
        self._model.compile(optimizer=self._optimizer,
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    def train(self, X, y, X_valid=None, y_valid=None, epochs=10, batch_size=16, lr=0.001, verbose=0):
        # compile the model
        optimizer = RMSprop(lr=lr)
        self._compile_model(optimizer)

        # train the model
        kwargs = {
            'batch_size': batch_size,
            'epochs': epochs,
            'verbose': verbose
        }
        if X_valid is not None and y_valid is not None:
            kwargs['validation_data'] = (X_valid, y_valid)

        history = self._model.fit(X, y, **kwargs)

        training_report = CNNKerasTrainingReport()
        training_report.loss = history.history['loss']
        training_report.val_loss = history.history['val_loss']
        training_report.acc = history.history['acc']
        training_report.val_acc = history.history['val_acc']

        return training_report

    def evaluate(self, X, y):
        loss, accuracy = self._model.evaluate(X, y)
        return {'loss': loss, 'accuracy': accuracy}

    def predict(self, X):
        return self._model.predict(X)

    def save(self, filepath):
        json_obj = {
            'conv_dropout': self._conv_dropout,
            'dense_dropout': self._dense_dropout,
            'config': self._model.get_config(),
            'weights': self._transform_weights_to_lists(self._model.get_weights())
        }
        
        with gzip.open(filepath, 'wb') as f:
            f.write(json.dumps(json_obj, ensure_ascii=False).encode('utf-8'))

        return None
