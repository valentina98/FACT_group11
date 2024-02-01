import torch
import torch.nn as nn
from tqdm import tqdm
from data import get_dataset
import argparse
from torch.optim import AdamW
from models import get_model
from torch.optim.lr_scheduler import ReduceLROnPlateau

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_name", default="clip:RN50", type=str, help="Model backbone")
    parser.add_argument("--dataset", default="cifar10_val", type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder for model/run info.")
    parser.add_argument("--lr", default=1e-8, type=float)
    return parser.parse_args()

class CustomClassifier(nn.Module):
    def __init__(self, backbone, num_labels):
        super(CustomClassifier, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.output_dim, num_labels)

    def forward(self, inputs):
        with torch.no_grad():
            features = self.backbone(inputs)
        features = features.to(dtype=self.classifier.weight.dtype)
        logits = self.classifier(features)
        return logits

def get_model_final(args, backbone,num_labels):
    if "clip" in args.backbone_name:
        backbone = backbone.visual
        backbone.output_dim = backbone.output_dim
    else:
        return

    for param in backbone.parameters():
        param.requires_grad = False

    model = CustomClassifier(backbone, num_labels)
    return model

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_loss = total_loss / total
    accuracy = correct / total
    return average_loss, accuracy

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_accuracy, model):
        score = val_accuracy
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def main(args):
    train_loader, val_loader, test_loader, classes = get_dataset(args)
    if "resnet18_cub" in args.backbone_name:
        model, backbone, preprocess = get_model(args, backbone_name=args.backbone_name, full_model=True)
    else:
        backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
        num_labels = len(classes)
        model = get_model_final(args, backbone, num_labels)

    model.to(args.device)

    train_loader, val_loader, test_loader, classes = get_dataset(args, preprocess)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1, verbose=True)

    early_stopping = EarlyStopping(patience=7, verbose=True, delta=0.01)

    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, args.device)

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, args.device)

        print(f"Epoch {epoch + 1}/{args.epochs}, Training Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        scheduler.step(val_loss)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    test_loss, test_accuracy = evaluate(model, test_loader, criterion, args.device)
    print(f"Final Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    args = config()
    main(args)
