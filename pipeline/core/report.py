from abc import ABC, abstractmethod
import json

import matplotlib.pyplot as plt


class Report(ABC):

    @classmethod
    def load(cls, filepath):
        return cls.load(filepath)

    def __init__(self):
        super().__init__()

    @abstractmethod
    def save(self, filepath):
        pass


class CNNKerasTrainingReport(Report):

    @staticmethod
    def load(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            json_obj = json.load(f)

        report = CNNKerasTrainingReport()
        report.loss = json_obj['loss']
        report.val_loss = json_obj['val_loss']
        report.acc = json_obj['acc']
        report.val_acc = json_obj['val_acc']

        return report

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

    def display_loss_curve(self):
        epochs = range(1, len(self.loss) + 1)
        plt.plot(epochs, self.loss, color='b', label='loss')
        plt.plot(epochs, self.val_loss, color='r', label='val_loss')
        plt.legend()
        plt.grid()
        plt.title('Training and validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
        return None

    def display_acc_curve(self):
        epochs = range(1, len(self.acc) + 1)
        plt.plot(epochs, self.acc, color='b', label='acc')
        plt.plot(epochs, self.val_acc, color='r', label='val_acc')
        plt.legend()
        plt.grid()
        plt.title('Training and validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()
        return None
