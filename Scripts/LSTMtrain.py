import sys
sys.path.append("C:\College\Projects\Multilabel-Movie-Genre-Classification")
from utils.models import LSTMbaseGloVe
from utils.train import TrainLoopCompact
from utils.nlp import PrepareData, TextDataset, get_word_to_index

import torch
import numpy as np
from torch.utils.data import DataLoader

def Train():
    model = LSTMbaseGloVe(embed_dim=200, embeddings='6B', hidden_dim=384)
    wti = get_word_to_index('.vector_cache/glove.6B.200d.txt')
    xtrain = np.load("Data/Training/x_train.npy", allow_pickle=True)
    ytrain = np.load("Data/Training/y_train.npy", allow_pickle=True)

    xval = np.load("Data/Validation/x_val.npy", allow_pickle=True)
    yval = np.load("Data/Validation/y_val.npy", allow_pickle=True)

    xtrain= PrepareData(xtrain)
    xval= PrepareData(xval)

    train_set = TextDataset(xtrain, ytrain, wti)
    val_set = TextDataset(xval, yval, wti)
    train_loader = DataLoader(train_set, 64, True)
    val_loader = DataLoader(val_set, 64, True)

    optimizer = torch.optim.AdamW(model.parameters(), 4e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def lr_lambda(epoch):
        if epoch <= 20:
            return 0.9
        return 1
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda)

    TrainLoopCompact(model, optimizer, loss_fn, train_loader, val_loader, scheduler, 50, 10, batch_loss=10)
    torch.save(model.state_dict(), 'Models/LSTM3200.pth')

if __name__ == '__main__':
    Train()