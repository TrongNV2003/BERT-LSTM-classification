# BERT-LSTM-classification
Predict multiple intents for each message on conversations, using RNN/LSTM to represent as "context history".

## Methodology
In this repo, my dataset has several dialogues, each dialogue has many turns. So i process dataloader in difference way.
There are 2 steps to tokenize. First, processing turns in a dialogue. Second, processing token in turns

1. Dataloader:
- The batch size for train, val and test is counted by dialogues (e.g. batch_size = 3 -> 3 dialogues per batch)

2. Dynamic padding:
- Select the dialogue has the longest turns in batch, then padding others dialogues to make sure that all dialogue has same turns. This mechanism make sure that it may not cut turns in dialogue if this dialgue is too long.

3. Reset hidden states:
- Since each dialogue is difference in context, so we have to reset hidden states after each dialogue to make sure that hidden state in new conversation does not update from the old one.


## Installation
```sh
pip install -r requirements.txt
```

## Usage
training and evaluating models:
```sh
bash train.sh
```
