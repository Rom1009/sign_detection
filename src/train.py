import time
from tqdm import tqdm
import torch
from src.models.accuracy import accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, data_loader, optimizer, loss_criteria, history_train):
    """
    Input:
    - model: the neural network model to train.
    - data_loader: DataLoader providing batches of training data and labels.
    - optimizer: the optimizer used for updating model parameters.
    - loss_criteria: the loss function to compute loss.
    - history_train: a dictionary to store training loss, accuracy, and time history.

    Output:
    - epoch_loss: average training loss for the epoch.
    - epoch_acc: average training accuracy for the epoch.
    - total_time: time taken to complete the epoch.

    Description:
    Trains the model for one epoch. It iterates over batches of data, calculates the predictions and loss,
    computes accuracy, and updates the model parameters. Training statistics are collected and returned.
    """
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()
    model.train()

    for batch_idx, (images, labels) in tqdm(enumerate(data_loader)):
        images = images.to(device)
        labels = labels.flatten().to(device)

        optimizer.zero_grad()

        pred = model(images)
        loss = loss_criteria(pred, labels)
        epoch_loss.append(loss.item())

        acc = accuracy(pred, labels)
        epoch_acc.append(acc)

        loss.backward()
        optimizer.step()

    end_time = time.time()
    total_time = end_time - start_time

    epoch_loss = torch.mean(torch.tensor(epoch_loss, dtype=torch.float32))
    epoch_acc = torch.mean(torch.tensor(epoch_acc, dtype=torch.float32))

    history_train["loss"].append(epoch_loss)
    history_train["acc"].append(epoch_acc)
    history_train["time"].append(total_time)

    return epoch_loss, epoch_acc, total_time


def test_one_epoch(model, data_loader, loss_criteria, history_test,
                   model_name, best_val_acc):
    """
    Input:
    - model: the neural network model to evaluate.
    - data_loader: DataLoader providing batches of validation data and labels.
    - loss_criteria: the loss function to compute loss.
    - history_test: a dictionary to store validation loss, accuracy, and time history.
    - model_name: name for saving the model if validation accuracy improves.
    - best_val_acc: the best validation accuracy observed so far.

    Output:
    - epoch_loss: average validation loss for the epoch.
    - epoch_acc: average validation accuracy for the epoch.
    - total_time: time taken to complete the epoch.

    Description:
    Evaluates the model on validation data for one epoch. It calculates predictions and loss,
    computes accuracy, and checks if the current validation accuracy is the best. Validation statistics 
    are collected and returned.
    """
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()
    model.eval()

    for batch_idx, (images, labels) in tqdm(enumerate(data_loader)):
        images = images.to(device)
        labels = labels.flatten().to(device)

        with torch.inference_mode():
            pred = model(images)

        loss = loss_criteria(pred, labels)
        epoch_loss.append(loss.item())

        acc = accuracy(pred, labels)
        epoch_acc.append(acc)

    end_time = time.time()
    total_time = end_time - start_time

    epoch_loss = torch.mean(torch.tensor(epoch_loss, dtype=torch.float32))
    epoch_acc = torch.mean(torch.tensor(epoch_acc, dtype=torch.float32))

    history_test["loss"].append(epoch_loss)
    history_test["acc"].append(epoch_acc)
    history_test["time"].append(total_time)

    if epoch_acc > best_val_acc:
        best_val_acc = epoch_acc
        torch.save(model.state_dict(), f"{model_name}.pth")

    return epoch_loss, epoch_acc, total_time