import torch
import csv
from sklearn.metrics import confusion_matrix
from src.models.model_c import AntiImpersonationModel
from src.data.model_c_dataset import AntiImpersonationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def eval_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AntiImpersonationModel().to(device)
    model.load_state_dict(torch.load('experiments/model_c_control/best_model.pt', map_location=device, weights_only=True))
    model.eval()

    val_ds = AntiImpersonationDataset('data/manifests/val_pairs.csv')
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, embs, labels in tqdm(val_loader):
            logits = model(imgs.to(device), embs.to(device))
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    print("Confusion Matrix:")
    print(cm)
    
    with open('experiments/model_c_control/confusion_matrix_raw.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['True \ Predicted', 'Predicted Genuine (0)', 'Predicted Different (1)', 'Predicted Impersonation (2)'])
        writer.writerow(['True Genuine (0)', cm[0][0], cm[0][1], cm[0][2]])
        writer.writerow(['True Different (1)', cm[1][0], cm[1][1], cm[1][2]])
        writer.writerow(['True Impersonation (2)', cm[2][0], cm[2][1], cm[2][2]])

if __name__ == "__main__":
    eval_model()
