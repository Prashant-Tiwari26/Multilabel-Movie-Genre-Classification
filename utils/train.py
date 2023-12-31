import torch
import numpy as np
import seaborn as sns
from tqdm.tk import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def TrainLoopCompact(
    model,
    optimizer:torch.optim.Optimizer,
    criterion:torch.nn.Module,
    train_dataloader:torch.utils.data.DataLoader,
    val_dataloader:torch.utils.data.DataLoader,
    scheduler=None,
    num_epochs:int=20,
    early_stopping_rounds:int=5,
    return_best_model:bool=True,
    batch_loss:int = 5,
    device:str='cuda'
):
    model.to(device)
    best_val_loss = float('inf')
    total_train_loss = []
    total_val_loss = []
    epochs_without_improvement = 0
    best_weights = model.state_dict()

    for epoch in tqdm(range(num_epochs), desc='Training Epochs'):
        model.train()
        print("\n---------------------\nEpoch {} | Learning Rate = {}".format(epoch, optimizer.param_groups[0]['lr']))
        train_loss = 0
        for i, (texts, labels) in tqdm(enumerate(train_dataloader), desc='Train Batch Counter', total=len(train_dataloader)):
            texts = texts.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            train_loss += loss
            loss.backward()
            optimizer.step()
            if i % batch_loss == 0:
                print("Loss for batch {} = {}".format(i, loss))

        print("\nTraining Loss for epoch {} = {}\n".format(epoch, train_loss))
        total_train_loss.append(train_loss/len(train_dataloader.dataset))

        model.eval()
        with torch.inference_mode():
            validation_loss = 0
            for (texts, labels) in val_dataloader:
                texts = texts.to(device)
                labels = labels.to(device).float()
                outputs = model(texts)
                loss = criterion(outputs, labels)
                validation_loss += loss

            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                epochs_without_improvement = 0
                best_weights = model.state_dict()
            else:
                epochs_without_improvement += 1

            print(f"Current Validation Loss = {validation_loss}")
            print(f"Best Validation Loss = {best_val_loss}")
            print(f"Epochs without Improvement = {epochs_without_improvement}")
        
        total_val_loss.append(validation_loss/len(val_dataloader.dataset))
        if scheduler is not None:
            scheduler.step()
        
        if epochs_without_improvement == early_stopping_rounds:
            print("Early stopping triggered")
            break

    if return_best_model == True:
        model.load_state_dict(best_weights)

    total_train_loss = [item.cpu().detach().numpy() for item in total_train_loss]
    total_val_loss = [item.cpu().detach().numpy() for item in total_val_loss]

    total_train_loss = np.array(total_train_loss)
    total_val_loss = np.array(total_val_loss)

    x_train = np.arange(len(total_train_loss))
    x_val = np.arange(len(total_val_loss))

    sns.set_style('whitegrid')
    plt.figure(figsize=(12,9))

    sns.lineplot(x=x_train, y=total_train_loss, label='Training Loss')
    sns.lineplot(x=x_val, y=total_val_loss, label='Validation Loss')
    plt.title("Loss over {} Epochs".format(len(total_train_loss)))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(np.arange(len(total_train_loss)))

    plt.show()

def TrainLoopSLM(
    model,
    optimizer:torch.optim.Optimizer,
    criterion:torch.nn.Module,
    train_dataloader:torch.utils.data.DataLoader,
    val_dataloader:torch.utils.data.DataLoader,
    scheduler=None,
    num_epochs:int=20,
    early_stopping_rounds:int=5,
    return_best_model:bool=True,
    batch_loss:int = 5,
    device:str='cuda'
):
    model.to(device)
    best_val_loss = float('inf')
    total_train_loss = []
    total_val_loss = []
    epochs_without_improvement = 0
    best_weights = model.state_dict()

    for epoch in tqdm(range(num_epochs), desc='Training Epochs'):
        model.train()
        print("\n---------------------\nEpoch {} | Learning Rate = {}".format(epoch, optimizer.param_groups[0]['lr']))
        train_loss = 0
        for i, batch in tqdm(enumerate(train_dataloader), desc='Train Batch Counter', total=len(train_dataloader)):
            tokens = batch['input_ids'].to(device, dtype=torch.long)
            masks = batch['attention_mask'].to(device, dtype=torch.long)
            labels = batch['targets'].to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(tokens, masks)
            loss = criterion(outputs, labels)
            train_loss += loss
            loss.backward()
            optimizer.step()
            if i % batch_loss == 0:
                print("\033[3mLoss for batch {} = {}\033[0m".format(i, loss))

        print("\nTraining Loss for epoch {} = {}\n".format(epoch, train_loss))
        total_train_loss.append(train_loss/len(train_dataloader.dataset))

        model.eval()
        with torch.inference_mode():
            validation_loss = 0
            for batch in val_dataloader:
                tokens = batch['input_ids'].to(device, dtype=torch.long)
                masks = batch['attention_mask'].to(device, dtype=torch.long)
                labels = batch['targets'].to(device, dtype=torch.float)
                outputs = model(tokens, masks)
                loss = criterion(outputs, labels)
                validation_loss += loss

            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                epochs_without_improvement = 0
                best_weights = model.state_dict()
            else:
                epochs_without_improvement += 1

            print(f"Current Validation Loss = {validation_loss}")
            print(f"Best Validation Loss = {best_val_loss}")
            print(f"Epochs without Improvement = {epochs_without_improvement}")
        
        total_val_loss.append(validation_loss/len(val_dataloader.dataset))
        if scheduler is not None:
            scheduler.step()
        
        if epochs_without_improvement == early_stopping_rounds:
            print("\033[1mEarly stopping triggered\033[0m")
            break

    if return_best_model == True:
        model.load_state_dict(best_weights)

    total_train_loss = [item.cpu().detach().numpy() for item in total_train_loss]
    total_val_loss = [item.cpu().detach().numpy() for item in total_val_loss]

    total_train_loss = np.array(total_train_loss)
    total_val_loss = np.array(total_val_loss)

    x_train = np.arange(len(total_train_loss))
    x_val = np.arange(len(total_val_loss))

    sns.set_style('whitegrid')
    plt.figure(figsize=(12,9))

    sns.lineplot(x=x_train, y=total_train_loss, label='Training Loss')
    sns.lineplot(x=x_val, y=total_val_loss, label='Validation Loss')
    plt.title("Loss over {} Epochs".format(len(total_train_loss)))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(np.arange(len(total_train_loss)))

    plt.show()