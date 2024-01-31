import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse

from data import get_dataset
from models import get_model

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--backbone_name", default="clip:RN50", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder for model/run info.")
    return parser.parse_args()

def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, test_loader, num_classes):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)

            if num_classes == 2:
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_preds.extend(probs.tolist())
            else:
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.tolist())

            all_labels.extend(labels.tolist())

    if num_classes == 2:
        return roc_auc_score(all_labels, all_preds)
    else:
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        return correct / len(all_labels)

def main(args):

    backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
    train_loader, test_loader, _, classes = get_dataset(args, preprocess)

    for param in backbone.parameters():
        param.requires_grad = False

    num_classes = len(classes)
    classifier = nn.Linear(backbone.output_dim, num_classes)
    model = nn.Sequential(backbone, classifier)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(classifier.parameters(), lr=0.001)

    num_epochs = args.epochs
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer)
        test_accuracy = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    args = config()
    main(args)