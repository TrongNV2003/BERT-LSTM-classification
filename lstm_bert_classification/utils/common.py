from enum import Enum

class ModelType(str, Enum):
    LSTM = "lstm"
    RNN = "rnn"