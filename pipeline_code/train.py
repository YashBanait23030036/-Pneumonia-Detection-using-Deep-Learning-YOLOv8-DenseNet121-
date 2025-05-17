import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from pipeline_code.evaluate import evaluate_model

def train_model(model, train_loader, val_loader, config):
    model.to(config.device)
    criterion = nn.CrossEntropyLoss(weight=config.class_weights.to(config.device))
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_f1 = 0.0

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for images, labels in pbar:
            images = images.to(config.device)
            labels = labels.to(config.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=running_loss/(total//config.batch_size), accuracy=100.*correct/total)

        val_metrics = evaluate_model(model, val_loader, config)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, "best_model.pth"))
            print("💾 Model checkpoint saved!")

    return model
