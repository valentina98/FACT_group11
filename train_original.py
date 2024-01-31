import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from data import get_dataset
import argparse
from torch.optim import AdamW
from models import get_model

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_name", default="clip:RN50", type=str, help="Model backbone")
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder for model/run info.")
    parser.add_argument("--lr", default=1e-2, type=float)
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

def evaluate(model, test_loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def main(args):
    if "resnet18_cub" in args.backbone_name:
        model, backbone, preprocess = get_model(args, backbone_name=args.backbone_name, full_model=True)
    else:
        backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
        train_loader, test_loader, _, classes = get_dataset(args, preprocess)
        num_labels = len(classes)
        model = get_model_final(args, backbone, num_labels)

    model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    train_loader, test_loader, _, classes = get_dataset(args, preprocess)

    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, args.device)
        accuracy = evaluate(model, test_loader, args.device)
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    accuracy = evaluate(model, test_loader, args.device)
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    args = config()
    main(args)