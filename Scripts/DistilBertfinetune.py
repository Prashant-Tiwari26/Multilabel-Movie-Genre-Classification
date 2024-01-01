import sys
sys.path.append("C:\College\Projects\Multilabel-Movie-Genre-Classification")
from utils.models import DistilBertBaseUncased
from utils.train import TrainLoopSLM
from utils.nlp import TextDatasetSLM

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer

def finetune():
    model = DistilBertBaseUncased()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    xtrain = np.load("Data/Training/x_train.npy", allow_pickle=True)
    ytrain = np.load("Data/Training/y_train.npy", allow_pickle=True)
    xval = np.load("Data/Validation/x_val.npy", allow_pickle=True)
    yval = np.load("Data/Validation/y_val.npy", allow_pickle=True)

    train_set = TextDatasetSLM(xtrain, ytrain, tokenizer)
    val_set = TextDatasetSLM(xval, yval, tokenizer)
    train_loader = DataLoader(train_set, 64, True)
    val_loader = DataLoader(val_set, 64, True)

    def lr_lambda(epoch):
        if epoch <= 20:
            return 0.9
        return 1
    optimizer = torch.optim.AdamW(model.parameters(), 3e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda)

    TrainLoopSLM(model, optimizer, loss_fn, train_loader, val_loader, scheduler, 50, 10, batch_loss=5)
    torch.save(model.state_dict(), 'Models/distilbert.pth')

if __name__ == '__main__':
    finetune()