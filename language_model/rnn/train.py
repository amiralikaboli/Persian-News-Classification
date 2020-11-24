import json
import math
from datetime import datetime

import torch
import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(self, num_chars, hidden_size, num_layers):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(num_chars, num_chars)
        self.lstm = nn.LSTM(input_size=num_chars, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, num_chars)

    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        output, hidden_state = self.lstm(embedding, hidden_state)
        output = self.decoder(output)
        return output, (hidden_state[0].detach(), hidden_state[1].detach())


if __name__ == '__main__':
    now_datetime = datetime.now()

    hyper_parameters = {
        "hidden_size": 256,
        "num_layers": 1,
        "lr": 0.002,
        "num_epochs": 100
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open('data/train_sentences.json', 'r', encoding='utf-8') as json_file:
        train_sentences = json.load(json_file)
    with open('data/char2index.json', 'r', encoding='utf-8') as json_file:
        char2index = json.load(json_file)
    with open('data/index2char.json', 'r', encoding='utf-8') as json_file:
        index2char = json.load(json_file)

    rnn = CharRNN(len(char2index), hyper_parameters["hidden_size"], hyper_parameters["num_layers"])
    rnn = rnn.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=hyper_parameters["lr"])

    best_training_loss = math.inf
    for epoch_ind in range(hyper_parameters["num_epochs"]):
        training_loss = 0
        hidden_state = None

        for indexed_sentence in train_sentences:
            indexed_sentence = torch.tensor([indexed_sentence]).to(device)
            indexed_sentence = torch.squeeze(indexed_sentence, dim=1).T
            input_seq = indexed_sentence[:-1]
            target_seq = indexed_sentence[1:]

            output, hidden_state = rnn(input_seq, hidden_state)

            loss = loss_func(torch.squeeze(output), torch.squeeze(target_seq))
            training_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch: {0} \t Loss: {1:.8f}".format(epoch_ind, training_loss / len(train_sentences)))
        if best_training_loss > training_loss:
            best_training_loss = training_loss
            torch.save(rnn.state_dict(), f'data/checkpoints/{now_datetime}.pth')
