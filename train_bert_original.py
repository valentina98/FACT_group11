import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
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
    parser.add_argument("--epochs", default=3, type=int)
    return parser.parse_args()

class BertLinearClassifier(nn.Module):
    def __init__(self, bert_model, num_labels):
        super(BertLinearClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[1]
        logits = self.classifier(sequence_output)
        return logits

def main(args, train_loader, test_loader, classes):
    device = args.device
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    num_labels = len(classes)
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = BertLinearClassifier(bert_model, num_labels)
    model.to(device)

    # Freeze BERT layers
    for param in model.bert.parameters():
        param.requires_grad = False

    optim = AdamW(model.classifier.parameters(), lr=5e-5)

    for epoch in range(args.epochs):
        model.train()
        for texts, labels in tqdm(train_loader):
            labels = labels.to(device)
            encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)

            logits = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optim.step()
            optim.zero_grad()

    # Evaluation
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for texts, labels in tqdm(test_loader):
            labels = labels.to(device)
            encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)

            logits = model(input_ids, attention_mask)
            predictions.extend(logits.argmax(dim=-1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    args = config()
    train_loader, test_loader, _, classes = get_dataset(args)
    main(args, train_loader, test_loader, classes)
