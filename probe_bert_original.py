import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score

# Presumed get_dataset method
# from data import get_dataset

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="20ng", type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    return parser.parse_args()

class BertLinearClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertLinearClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = outputs.pooler_output
        logits = self.classifier(sequence_output)
        return logits

def main(args):
    train_loader, val_loader, test_loader, classes = get_dataset(args)

    device = torch.device(args.device)
    model = BertLinearClassifier(num_labels=len(classes)).to(device)

    optimizer = AdamW(model.classifier.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1, verbose=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for epoch in range(args.epochs):
        model.train()
        for texts, labels in tqdm(train_loader):
            labels = labels.to(device)
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for texts, labels in tqdm(val_loader):
                labels = labels.to(device)
                inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']

                logits = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(logits, labels)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss}")
        scheduler.step(avg_val_loss)

    # Test phase
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for texts, labels in tqdm(test_loader):
            labels = labels.to(device)
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            logits = model(input_ids, attention_mask)
            predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    print(f"Test Accuracy: {accuracy}")

if __name__ == "__main__":
    args = config()
    main(args)