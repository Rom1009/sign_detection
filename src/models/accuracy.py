import torch 

def accuracy(pred, label):
    """
    Input:
    - pred: tensor of shape [B, num_classes], predicted class scores for B samples.
    - label: tensor of shape [B], true labels for B samples.

    Output:
    - acc: float, the accuracy percentage of the predictions.

    Description: 
    Calculates the accuracy of predicted class labels against true labels.
    The function uses `torch.argmax` to find the predicted class with the highest score for each sample.
    It then compares these predictions with the true labels to count the number of correct predictions.
    Finally, it computes the accuracy as the ratio of correct predictions to the total number of samples, 
    multiplied by 100 to express it as a percentage.
    """
    # Get the predicted labels by finding the index of the maximum value along dimension 1
    pred = torch.argmax(pred, dim=1)
    # Compare predicted labels with actual labels
    correct = (pred == label).sum().item()
    # Calculate accuracy
    acc = correct / len(label) * 100
    return acc