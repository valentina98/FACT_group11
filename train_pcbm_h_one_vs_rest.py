import argparse
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
import sys
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import softmax
from sklearn.metrics import average_precision_score

from data import get_dataset
from concepts import ConceptBank
from models import PosthocLinearCBM, PosthocHybridCBM, get_model
from training_tools import load_or_compute_projections, AverageMeter, MetricComputer



def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder")
    parser.add_argument("--pcbm-path", required=True, type=str, help="Trained PCBM module.")
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank.")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--dataset", default="cub", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--l2-penalty", default=0.01, type=float)
    parser.add_argument("--num-workers", default=2, type=int)
    return parser.parse_args()


@torch.no_grad()
def eval_model(args, posthoc_layer, loader, num_classes):
    for class_idx in range(num_classes):
            
        epoch_summary = {"Accuracy": AverageMeter()}
        tqdm_loader = tqdm(loader)
        computer = MetricComputer(n_classes=num_classes)
        all_preds = []
        all_labels = []
        
        for batch_X, batch_Y in tqdm(loader):
            batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device) 
            out = posthoc_layer(batch_X)            
            all_preds.append(out.detach().cpu().numpy())
            all_labels.append(batch_Y.detach().cpu().numpy())
            metrics = computer(out, batch_Y) 
            epoch_summary["Accuracy"].update(metrics["accuracy"], batch_X.shape[0])
            epoch_summary["Precision"].update(metrics["precision"], batch_X.shape[0])

            summary_text = [f"Avg. {k}: {v.avg:.3f}" for k, v in epoch_summary.items()]
            summary_text = "Eval - " + " ".join(summary_text)
            tqdm_loader.set_description(summary_text)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate average precision for each class
    average_precisions = []
    for class_idx in range(num_classes):
        class_preds = softmax(all_preds, axis=1)[:, class_idx]
        binary_labels = (all_labels == class_idx).astype(int)
        average_precisions.append(average_precision_score(binary_labels, class_preds))

    # Compute Mean Average Precision
    mean_avg_precision = np.mean(average_precisions)

    # Calculate overall accuracy
    overall_accuracy = np.mean(all_labels == np.argmax(all_preds, axis=1))

    return {"Accuracy": overall_accuracy, "mAP": mean_avg_precision}


def train_hybrid(args, train_loader, val_loader, posthoc_layer, optimizer, num_classes):
    cls_criterion = nn.CrossEntropyLoss()
    average_precisions = []

    for epoch in range(1, args.num_epochs+1):
        print(f"Epoch: {epoch}")

        for class_idx in range(num_classes):
            epoch_summary = {"CELoss": AverageMeter(), "Accuracy": AverageMeter(), "Precision": AverageMeter()}
            tqdm_loader = tqdm(train_loader)
                
            computer = MetricComputer(n_classes=num_classes)

            for batch_X, batch_Y in tqdm(train_loader):
                batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device)
                
                # Create binary labels for current class vs all others
                binary_labels = (batch_Y == class_idx).long()

                optimizer.zero_grad()
                out, projections = posthoc_layer(batch_X, return_dist=True)
                cls_loss = cls_criterion(out, binary_labels)
                loss = cls_loss + args.l2_penalty*(posthoc_layer.residual_classifier.weight**2).mean()
                loss.backward()
                optimizer.step()
                
                epoch_summary["CELoss"].update(cls_loss.detach().item(), batch_X.shape[0])
                metrics = computer(out, batch_Y) 
                epoch_summary["Accuracy"].update(metrics["accuracy"], batch_X.shape[0])
                epoch_summary["Precision"].update(metrics["precision"], batch_X.shape[0])

                summary_text = [f"Avg. {k}: {v.avg:.3f}" for k, v in epoch_summary.items()]
                summary_text = " ".join(summary_text)
                tqdm_loader.set_description(summary_text) 
            
            ap = epoch_summary["Precision"].avg
            average_precisions.append(ap)

        mean_avg_precision = sum(average_precisions) / num_classes
        print("Mean Average Precision: ", mean_avg_precision)

        latest_info = dict()
        latest_info["epoch"] = epoch
        latest_info["args"] = args
        latest_info["train_acc"] = epoch_summary["Accuracy"]
        latest_info["test_acc"] = eval_model(args, posthoc_layer, val_loader, num_classes)["Accuracy"]
        latest_info["train_map"] = mean_avg_precision
        latest_info["test_map"] = eval_model(args, posthoc_layer, val_loader, num_classes)["mAP"]

        print("Final test accuracy: ", latest_info["test_acc"].avg)
        print("Final test mean average precision: ", latest_info["test_map"].avg)

    return latest_info



def main(args, backbone, preprocess):
    train_loader, test_loader, idx_to_class, classes = get_dataset(args, preprocess)
    num_classes = len(classes)
    
    hybrid_model_path = args.pcbm_path.replace("pcbm_", "pcbm-hybrid_")
    run_info_file = hybrid_model_path.replace("pcbm", "run_info-pcbm")
    run_info_file = run_info_file.replace(".ckpt", ".pkl")
    
    run_info_file = os.path.join(args.out_dir, run_info_file)
    
    # We use the precomputed embeddings and projections.
    train_embs, _, train_lbls, test_embs, _, test_lbls = load_or_compute_projections(args, backbone, posthoc_layer, train_loader, test_loader)

    
    train_loader = DataLoader(TensorDataset(torch.tensor(train_embs).float(), torch.tensor(train_lbls).long()), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_embs).float(), torch.tensor(test_lbls).long()), batch_size=args.batch_size, shuffle=False)

    # Initialize PCBM-h
    hybrid_model = PosthocHybridCBM(posthoc_layer)
    hybrid_model = hybrid_model.to(args.device)
    
    # Initialize the optimizer
    hybrid_optimizer = torch.optim.Adam(hybrid_model.residual_classifier.parameters(), lr=args.lr)
    hybrid_model.residual_classifier = hybrid_model.residual_classifier.float()
    hybrid_model.bottleneck = hybrid_model.bottleneck.float()
    
    # Train PCBM-h
    run_info = train_hybrid(args, train_loader, test_loader, hybrid_model, hybrid_optimizer, num_classes)

    torch.save(hybrid_model, hybrid_model_path)
    with open(run_info_file, "wb") as f:
        pickle.dump(run_info, f)
    
    print(f"Saved to {hybrid_model_path}, {run_info_file}")

if __name__ == "__main__":    
    args = config()    
    # Load the PCBM
    posthoc_layer = torch.load(args.pcbm_path)
    posthoc_layer = posthoc_layer.eval()
    args.backbone_name = posthoc_layer.backbone_name
    backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
    backbone = backbone.to(args.device)
    backbone.eval()
    main(args, backbone, preprocess)
