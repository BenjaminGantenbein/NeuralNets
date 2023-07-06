import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)
from sklearn.preprocessing import label_binarize

device = 'cpu'
def data_loader(test_inputs, test_labels, batch_size=50):
    

    test_inputs, test_labels = torch.tensor(test_inputs), torch.tensor(test_labels)

  
    test_data = TensorDataset(test_inputs, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return test_dataloader

def evaluate_model(model, test_inputs, test_labels, loss_fn):
    
    test_dataloader = data_loader(test_inputs, test_labels)
    model.eval()

   
    total_loss = 0.0
    total_correct = 0

   
    all_preds = []
    all_labels = []
    for batch in test_dataloader:
      
        b_input_ids, b_labels = tuple(t.to(device) for t in batch)

        with torch.no_grad():
           
            logits = model(b_input_ids)

         
            loss = loss_fn(logits, b_labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, axis=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(b_labels.cpu().tolist())
            total_correct += torch.sum(preds == b_labels).item()

    avg_loss = total_loss / len(test_dataloader)
    accuracy = total_correct / len(test_dataloader.dataset)
    confusion_mat = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    f1_score_val = f1_score(all_labels, all_preds, average='macro')
    precision_val = precision_score(all_labels, all_preds, average='macro')
    recall_val = recall_score(all_labels, all_preds, average='macro')
    auc_val = roc_auc_score(label_binarize(all_labels, classes=list(set(all_labels))), 
                            label_binarize(all_preds, classes=list(set(all_labels))), 
                            multi_class='ovo')
    
    print(f"Average test loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision_val:.4f}, Recall: {recall_val:.4f}, F1-score: {f1_score_val:.4f}, AUC-ROC: {auc_val:.4f}")
    print(f"Confusion Matrix: \n{confusion_mat}")
    print(f"Classification Report: \n{report}")

    return
