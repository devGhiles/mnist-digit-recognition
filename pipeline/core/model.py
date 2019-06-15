from abc import ABC, abstractmethod


class Model(ABC):

    @classmethod
    @abstractmethod
    def load(cls, file_path):
        pass

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
