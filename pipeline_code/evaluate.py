import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_model(model, val_loader, config):
    model.eval()
    model.to(config.device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(config.device)
            labels = labels.to(config.device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    print(f"🧪 Validation - Acc: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    return {"acc": acc, "f1": f1, "precision": precision, "recall": recall}
