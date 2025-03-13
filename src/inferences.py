from src.train import test_one_epoch, train_one_epoch

def run_model(model, train_loader, val_loader, loss_criteria, optimizer,
              history_train, history_test, best_val_acc, model_name, num_epochs):
    """
    Input:
    - model: the neural network model to train and evaluate.
    - train_loader: DataLoader providing batches of training data.
    - val_loader: DataLoader providing batches of validation data.
    - loss_criteria: the loss function to compute loss.
    - optimizer: the optimizer used for updating model parameters.
    - history_train: a dictionary to store training loss, accuracy, and time history.
    - history_test: a dictionary to store validation loss, accuracy, and time history.
    - best_val_acc: the best validation accuracy observed so far.
    - model_name: name for saving the model if validation accuracy improves.
    - num_epochs: total number of epochs to train the model.

    Output:
    - None: The function runs training and validation for specified epochs and logs results.

    Description:
    Manages the training and validation process over multiple epochs. It prints training and validation 
    statistics for each epoch, updating the history and saving the model if validation accuracy improves.
    """
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        print("\nTraining........")
        train_loss, train_acc, train_time = train_one_epoch(model,
                                                            train_loader,
                                                            optimizer,
                                                            loss_criteria,
                                                            history_train)
        print(f"\nEpoch {epoch + 1}")
        print(f"Train Loss: {train_loss}")
        print(f"Train Accuracy: {train_acc}")
        print(f"Train Time: {train_time}")

        print("\nValidating........")
        val_loss, val_acc, val_time = test_one_epoch(model, val_loader,
                                                     loss_criteria,
                                                     history_test,
                                                     model_name,
                                                     best_val_acc)
        print(f"\nEpoch {epoch + 1}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Accuracy: {val_acc}")
        print(f"Val Time: {val_time}")
