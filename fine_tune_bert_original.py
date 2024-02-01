import torch
from data import get_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import get_dataset

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="20ng_val", type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--lr", default=5e-3, type=float)
    return parser.parse_args()

def main(args):
    train_loader, test_loader, classes, val_loader = get_dataset(args)

    device = torch.device(args.device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(classes))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            texts, labels = batch
            labels = labels.to(device)
            optimizer.zero_grad()

            inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {key: val.to(device) for key, val in inputs.items()}
            outputs = model(**inputs, labels=labels)

            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epochs} - Training Loss: {avg_loss}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                texts, labels = batch
                labels = labels.to(device)

                inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
                inputs = {key: val.to(device) for key, val in inputs.items()}
                outputs = model(**inputs, labels=labels)

                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{args.epochs} - Validation Loss: {avg_val_loss}")

        scheduler.step(avg_val_loss)

    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            texts, labels = batch
            labels = labels.to(device)

            inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {key: val.to(device) for key, val in inputs.items()}
            outputs = model(**inputs)

            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    print(f"Test Accuracy: {accuracy}")

if __name__ == "__main__":
    args = config()
    main(args)