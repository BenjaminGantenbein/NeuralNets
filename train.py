import time
from torch import nn
import torch
from evaluate import evaluate
import csv

if torch.cuda.is_available():
    device ='cuda'
else:
    device ='cpu'




def train(model, optimizer, train_dataloader, val_dataloader=None, epochs=10, loss_fn=None):
    """Train the CNN model."""
    with open('graph_data.csv', mode='a') as file:
        writer = csv.writer(file)

        # Tracking best validation accuracy
        best_accuracy = 0
        writer.writerow(['train_loss', 'val_loss', 'val_accuracy'])
    

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
                val_loss, val_accuracy = evaluate(model, val_dataloader, loss_fn)

                # Track the best accuracy
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save(model.state_dict(), 'best_model.pth')

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch
                writer.writerow([avg_train_loss, val_loss, val_accuracy])
                print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                
        print("\n")
        print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")
        