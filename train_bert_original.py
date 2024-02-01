import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from data import get_dataset
import argparse
from torch.optim import AdamW

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="20ng", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    return parser.parse_args()

def main(args,train_loader, test_loader, classes):

    device = args.device

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    num_labels = len(classes)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    model.to(device)

    optim = AdamW(model.parameters(), lr=args.lr)

    epochs = args.epochs

    for epoch in range(epochs):  # You can adjust the number of epochs
        model.train()
        for texts, labels in tqdm(train_loader):
            labels = labels.to(device)

            encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()
            optim.zero_grad()

    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for texts, labels in tqdm(test_loader):
            labels = labels.to(device)

            # Tokenize the texts in the batch
            encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(logits.argmax(dim=-1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            accuracy = accuracy_score(true_labels, predictions)
            print(f"Accuracy: {accuracy}")

    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    args = config()

    train_loader, test_loader, _, classes = get_dataset(args)
    num_labels = len(train_loader.dataset.labels.unique())
    accuracy = main(args,train_loader, test_loader, classes)
    print(f"Accuracy: {accuracy}")