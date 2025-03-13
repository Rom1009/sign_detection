import torch
import torch.nn as nn
from src.models.model import CNN1DTransformer
from src.inferences import run_model
from src.data_processing.data_loader import data_loader
'''
    Description:
    This code initializes the model training process, including setting the device, defining the optimizer, 
    loss criteria, learning rate scheduler, and running the training and validation process across multiple folds.
'''

train_data_loader, valid_data_loader = data_loader()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN1DTransformer(num_classes=7, seq_len=103, dim=192, num_heads=4)

# Move the model to the specified device (CPU or GPU)
model.to(device)

# Initialize best validation accuracy to track improvements
best_val_acc = 0

# Define the number of epochs for training
epoch = 10

# Dictionaries to store training and validation history
history_train = {"loss": [], "acc": [], "time": []}
history_test = {"loss": [], "acc": [], "time": []}

# Define the loss criteria with label smoothing to help prevent overfitting
loss_criteria = nn.CrossEntropyLoss(label_smoothing=0.1)

# Initialize the optimizer; Adam is used here with a learning rate and weight decay for regularization
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

# Set up a learning rate scheduler that reduces the learning rate by a factor of 0.5 every 5 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Loop over each fold for cross-validation
    
def run():
        # Run the model training and validation for the current fold
    run_model(model, train_data_loader, valid_data_loader, loss_criteria,
            optimizer, history_train, history_test, best_val_acc, "./models/CNN1D", epoch)
