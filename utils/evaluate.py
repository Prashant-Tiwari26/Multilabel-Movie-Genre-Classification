import sys
sys.path.append("C:\College\Projects\Multilabel-Movie-Genre-Classification")

import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.nlp import TextDataset, PrepareData
from sklearn.metrics import accuracy_score, classification_report

def evaluate_rnn_performance(xtest:np.ndarray, ytest:np.ndarray, word_to_index:dict, model:torch.nn.Module, batch_size:int=64, device:str='cuda'):
    xtest = PrepareData(xtest)
    test_loader = DataLoader(TextDataset(xtest, ytest, word_to_index), batch_size, True)
    true_labels = []
    pred_labels = []
    model.to(device)
    with torch.inference_mode():
        for texts, labels in test_loader:
            texts = texts.to(device)
            labels = labels.to(device).float()
            logits = model(texts)
            preds = torch.round(torch.sigmoid(logits))
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    accuracy = accuracy_score(true_labels, pred_labels)
    print(classification_report(true_labels, pred_labels))
    print(f'Accuracy: {accuracy * 100:.2f}%')
    