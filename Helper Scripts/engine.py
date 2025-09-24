
'''
Contains functions for training and testing a PyTorch model
'''

from typing import Dict, List, Tuple
from tqdm.auto import tqdm

import torch
import torch.nn as nn

def train_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              device = torch.device) -> Tuple[float,float]:

    #Putting the model on training mode
    model.train()

    #Setup train loss and train accuracy values
    train_loss, train_acc = 0,0

    #Looping through data batches
    for batch, (X,y) in enumerate(dataloader):
        
        #Moving data to target device
        X, y = X.to(device), y.to(device)

        #1.Forward pass
        y_pred = model(X)

        #2. Calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        #3. Zeroing the gradients
        optimizer.zero_grad()

        #4. Backprop
        loss.backward()

        #5. Optimizer step
        optimizer.step()

        #Calculate accuracy metric
        y_pred_class = torch.argmax(y_pred, dim = 1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    #Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss/len(dataloader)
    train_acc = train_acc/len(dataloader)

    return train_loss, train_acc


def test_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device = torch.device)->Tuple[float,float]:

    #putting model on evaluation mode
    model.eval()

    #Setup test loss and test accuracy values
    test_loss, test_acc = 0,0

    #turning on inference mode (we don't need the gradient engine activated)
    with torch.inference_mode():
        #Loop through Dataloader batches
        for batch, (X,y) in enumerate(dataloader):
            X,y = X.to(device), y.to(device)

            #1. Forward pass
            test_pred_logits = model(X)

            #2. Calculate the loss
            loss = loss_fn(test_pred_logits,y)
            test_loss += loss.item()

            #Calculate the accuracy
            test_pred_labels = torch.argmax(test_pred_logits, axis = 1)
            test_acc += (test_pred_labels==y).sum().item()/len(test_pred_labels)

    test_loss = test_loss/len(dataloader)
    test_acc = test_acc/len(dataloader)

    return test_loss, test_acc

def train(model: torch.nn.Module,
         train_dataloader,
         test_dataloader,
         optimizer,
         loss_fn: torch.nn.Module = nn.CrossEntropyLoss,
         epochs: int = 5,
         device = torch.device)->Dict[str,List]:

    #Create dictionary of lists
    results = {"train_loss": [],
              "train_acc": [],
              "test_loss": [],
              "test_acc": []}

    #looping through training and test steps
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model = model,
                                          dataloader = train_dataloader,
                                          loss_fn = loss_fn,
                                          optimizer = optimizer,
                                          device = device)

        test_loss, test_acc = test_step(model = model,
                                       dataloader = test_dataloader,
                                       loss_fn = loss_fn,
                                       device = device)

        #print out what's happening
        print(f'''Epoch: {epoch+1}| Train loss: {train_loss:.4f} and Train acuuracry: {train_acc:.4f} |
                  Test loss: {test_loss:.4f} and Test acc: {test_acc:.4f}''')


        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
