import json

from abc import ABC, abstractmethod


class TrainingReport(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def save(self, filepath):
        pass

    @abstractmethod
    def load(self, filepath):
        pass


class CNNKerasTrainingReport(TrainingReport):

    def __init__(self):
        super().__init__()
        self.loss = None
        self.val_loss = None
        self.acc = None
        self.val_acc = None

    def save(self, filepath):
        json_obj = {
            'loss': self.loss,
            'val_loss': self.val_loss,
            'acc': self.acc,
            'val_acc': self.val_acc
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_obj, f, ensure_ascii=False)

        return None

    def load(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json_obj = json.load(f)

        self.loss = json_obj['loss']
        self.val_loss = json_obj['val_loss']
        self.acc = json_obj['acc']
        self.val_acc = json_obj['val_acc']

        return None
