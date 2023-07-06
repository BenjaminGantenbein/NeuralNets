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
   
    with open('graph_data.csv', mode='a') as file:
        writer = csv.writer(file)

       
        best_accuracy = 0
        writer.writerow(['train_loss', 'val_loss', 'val_accuracy'])
    

       
        print("Start training...\n")
        print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*60)

        for epoch_i in range(epochs):
           

            t0_epoch = time.time()
            total_loss = 0

           
            model.train()

            for step, batch in enumerate(train_dataloader):
               
                b_input_ids, b_labels = tuple(t.to(device) for t in batch)

                model.zero_grad()

                logits = model(b_input_ids)

                loss = loss_fn(logits, b_labels)
                total_loss += loss.item()

                loss.backward()

                optimizer.step()

            avg_train_loss = total_loss / len(train_dataloader)
           
           
            if val_dataloader is not None:
               
                val_loss, val_accuracy = evaluate(model, val_dataloader, loss_fn)

              
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save(model.state_dict(), 'best_model.pth')

                time_elapsed = time.time() - t0_epoch
                writer.writerow([avg_train_loss, val_loss, val_accuracy])
                print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                
        print("\n")
        print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")
        