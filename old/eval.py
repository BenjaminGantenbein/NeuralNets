from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import torch
import numpy as np




def evaluate(model, dataloader):

    if torch.cuda.is_available():
        device = torch.device("cuda")
   
    else:
        device = torch.device("cpu")
    print("Eval Using : ", device)


    pred_y = []
    true_y = []

    model.eval()
    for batch in dataloader:
        
        X_batch, y_batch = batch
       
        X_batch = X_batch.to(device=device, dtype=torch.float)
        y_batch = y_batch.to(device=device, dtype=torch.float)

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred_batch = model(X_batch)

        pred_y.append(y_pred_batch.detach().cpu().numpy())
        true_y.append(y_batch.detach().cpu().numpy())

    pred_y = np.concatenate(pred_y, axis=0)
    pred_y = [1 if x > 0.5 else 0 for x in pred_y]
    y_true = np.concatenate(true_y, axis=0)

   
    print(confusion_matrix(y_true, pred_y))
    print('\n')
    print(classification_report(y_true, pred_y))

