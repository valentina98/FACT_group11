import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from data import get_dataset

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="20ng", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    return parser.parse_args()

def main(args,train_loader, test_loader, num_labels, device, epochs=10):
    # Initialize model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    model.to(device)
    optim = AdamW(model.parameters(), lr=5e-5)

    # Training
    model.train()
    for epoch in range(epochs):
        for batch in tqdm(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()

    # Evaluation
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            predictions.extend(logits.argmax(dim=-1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

if __name__ == "__main__":
    args = config
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_loader, test_loader, _, _ = get_dataset(args)
    num_labels = len(train_loader.dataset.labels.unique())
    accuracy = main(args,train_loader, test_loader, num_labels, device)
    print(f"Accuracy: {accuracy}")