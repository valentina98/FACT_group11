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
from sklearn.metrics import roc_auc_score

from data import get_dataset
from concepts import ConceptBank
from models import PosthocLinearCBM, PosthocHybridCBM, get_model
from training_tools import load_or_compute_projections, AverageMeter, MetricComputerMAP



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
    epoch_summary = {"mAP": AverageMeter()}
    tqdm_loader = tqdm(loader)
    computer = MetricComputerMAP(n_classes=num_classes)
    all_preds = []
    all_labels = []
    
    for batch_X, batch_Y in tqdm_loader:
        batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device) 
        out = posthoc_layer(batch_X)            
        all_preds.append(out.detach().cpu().numpy())
        all_labels.append(batch_Y.detach().cpu().numpy())
        metrics = computer(out, batch_Y) 
        epoch_summary["mAP"].update(metrics["mean_average_precision"], batch_X.shape[0])
        summary_text = [f"Avg. {k}: {v.avg:.3f}" for k, v in epoch_summary.items()]
        tqdm_loader.set_description("Eval - " + " ".join(summary_text))
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    if all_labels.max() == 1:
        auc = roc_auc_score(all_labels, softmax(all_preds, axis=1)[:, 1])
        epoch_summary["AUC"] = auc
        return epoch_summary
    
    return epoch_summary


def train_hybrid(args, train_loader, val_loader, posthoc_layer, optimizer, num_classes):
    cls_criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for multi-label classification
    for epoch in range(1, args.num_epochs+1):
        print(f"Epoch: {epoch}")
        epoch_summary = {"CELoss": AverageMeter(), "mAP": AverageMeter()}
        tqdm_loader = tqdm(train_loader)
        computer = MetricComputerMAP(n_classes=num_classes)
        all_train_preds = []
        all_train_labels = []
        for batch_X, batch_Y in tqdm_loader:
            batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device)
            optimizer.zero_grad()
            out, projections = posthoc_layer(batch_X, return_dist=True)
            
            # ??
            batch_Y = batch_Y.float()

            cls_loss = cls_criterion(out, batch_Y)
            loss = cls_loss + args.l2_penalty*(posthoc_layer.residual_classifier.weight**2).mean()
            loss.backward()
            optimizer.step()

            epoch_summary["CELoss"].update(cls_loss.detach().item(), batch_X.shape[0])

            all_train_preds.append(out.detach().cpu())
            all_train_labels.append(batch_Y.detach().cpu())

        all_train_preds = torch.cat(all_train_preds)
        all_train_labels = torch.cat(all_train_labels)
        train_mAP = computer._mean_average_precision(all_train_preds, all_train_labels)
        epoch_summary["mAP"].update(train_mAP)

        summary_text = [f"Avg. {k}: {v.avg:.3f}" for k, v in epoch_summary.items()]
        tqdm_loader.set_description("Train - " + " ".join(summary_text))

        latest_info = {
            "epoch": epoch,
            "args": args,
            "train_metrics": epoch_summary,
            "test_metrics": eval_model(args, posthoc_layer, val_loader, num_classes)
        }
        print("Final test metrics: ", latest_info["test_metrics"]["mAP"].avg)

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
