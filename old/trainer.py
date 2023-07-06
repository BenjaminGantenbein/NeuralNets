import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import time

if torch.cuda.is_available():
    device = torch.device("cuda")
   
else:
    device = torch.device("cpu")
print("Training Using : ", device)



def evaluate(model, val_dataloader, loss_fn):
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


def train(model, epochs, optimizer, loss_fn, train_loader, valid_loader):
    mean_train_losses = []
    mean_valid_losses = []
    valid_acc_list = []
    patience = 0
    best_score = 0

    for epoch in range(epochs):
        # switch the model to train mode
        model.train()
        
        train_losses = []
        valid_losses = []
        for i, batch in enumerate(train_loader):
            b_input_ids, b_labels = tuple(t.to(device) for t in batch)
            optimizer.zero_grad()
            outputs = model(b_input_ids)
            loss = loss_fn(outputs, b_labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
                
        model.eval()
        pred_labels = []
        true_labels = []
        with torch.no_grad():
            for i, (Xs, labels) in enumerate(valid_loader):
               
                outputs = model(Xs.float().to(device))
                loss = loss_fn(outputs.view(-1), labels.float().to(device))
                
                valid_losses.append(loss.item())
                
                predicted = [1 if d > 0.5 else 0 for d in outputs.data.squeeze()]
                pred_labels.extend(predicted)
                true_labels.extend(list(labels))
                
        mean_train_losses.append(np.mean(train_losses))
        mean_valid_losses.append(np.mean(valid_losses))
        
        accuracy = accuracy_score(true_labels, pred_labels)
        if accuracy > best_score:
            torch.save(model, 'best.pt')
            best_score = accuracy
            patience = 0 # reset patience
        else:
            patience += 1
            valid_acc_list.append(accuracy)
            print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}, valid acc : {:.2f}%, patience: {}'\
                .format(epoch+1, np.mean(train_losses), np.mean(valid_losses), accuracy, patience))
        if patience > 10:
            print('epoch : {}, patience : {}, training early stops'.format(epoch+1, patience))
            break

def train_two(model, optimizer, loss_fn, train_dataloader, val_dataloader=None, epochs=10):
 
    #Tracking best validation accuracy
    best_accuracy = 0
   

    # Start training loop
    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
    print("-"*60)

    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================

        # Tracking time and loss
        t0_epoch = time.time()
        total_loss = 0

        # Put the model into the training mode
        model.train()

        for step, batch in enumerate(train_dataloader):
            # Load batch to GPU
            b_input_ids, b_labels = tuple(t.to(device) for t in batch)
    

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Update parameters
            optimizer.step()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        # =======================================
        #               Evaluation
        # =======================================
        if val_dataloader is not None:
            # After the completion of each training epoch, measure the model's
            # performance on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader, loss_fn=loss_fn)

            # Track the best accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            
    print("\n")
    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")