import argparse
import os
import pickle
import numpy as np
import torch
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score

from data import get_dataset
from concepts import ConceptBank
from models import PosthocLinearMultilabelCBM, get_model
from training_tools import load_or_compute_projections


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder for model/run info.")
    parser.add_argument("--dataset", default="cub", type=str)
    parser.add_argument("--backbone-name", default="resnet18_cub", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lam", default=1e-5, type=float, help="Regularization strength.")
    parser.add_argument("--lr", default=1e-3, type=float)
    return parser.parse_args()

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import average_precision_score

def analyze_classifier(classifier, feature_names, k=5):
    """
    Analyzes the classifier to identify the top-k most influential features for each class.

    Args:
    - classifier: Trained multi-label classifier.
    - feature_names: List of feature (or concept) names.
    - k (int): Number of top features to select.

    Returns:
    - analysis (dict): Dictionary where keys are class indices and values are lists of top-k features.
    """
    analysis = {}
    for class_idx, estimator in enumerate(classifier.estimators_):
        # Get the feature weights for the current class
        class_weights = estimator.coef_[0]
        
        # Rank features based on the absolute weights
        top_features_indices = np.argsort(np.abs(class_weights))[-k:][::-1]
        
        # Get the feature names
        top_features = [feature_names[i] for i in top_features_indices]
        
        # Store the result
        analysis[class_idx] = top_features

    return analysis

def run_linear_probe(args, train_data, test_data):
    train_features, train_labels = train_data
    test_features, test_labels = test_data
    
    print("Train labels shape:", np.array(train_labels).shape)
    print("Test labels shape:", np.array(test_labels).shape)

    # Use OneVsRestClassifier for multi-label tasks
    classifier = OneVsRestClassifier(SGDClassifier(
        random_state=args.seed, loss="log_loss",
        alpha=args.lam, l1_ratio=args.alpha, verbose=0,
        penalty="elasticnet", max_iter=10000
    ))
    classifier.fit(train_features, train_labels)

    # Get probabilities for each class
    train_probabilities = classifier.predict_proba(train_features)
    test_probabilities = classifier.predict_proba(test_features)


    print("Train probabilities shape:", train_probabilities.shape)
    print("Test probabilities shape:", test_probabilities.shape)

    # Calculate mAP
    train_mAP = average_precision_score(train_labels, train_probabilities, average="macro")
    test_mAP = average_precision_score(test_labels, test_probabilities, average="macro")

    run_info = {
        "train_mAP": train_mAP,
        "test_mAP": test_mAP
    }

    # Access the coefficients and intercepts
    coefs = np.array([est.coef_ for est in classifier.estimators_])
    intercepts = np.array([est.intercept_ for est in classifier.estimators_])

    print("coefs shape:", coefs.shape)
    print("intercepts shape:", intercepts.shape)

    return run_info, coefs, intercepts

def main(args, concept_bank, backbone, preprocess):
    train_loader, test_loader, idx_to_class, classes = get_dataset(args, preprocess)
    
    # Get a clean conceptbank string
    # e.g. if the path is /../../cub_resnet-cub_0.1_100.pkl, then the conceptbank string is resnet-cub_0.1_100
    # which means a bank learned with 100 samples per concept with C=0.1 regularization parameter for the SVM. 
    # See `learn_concepts_dataset.py` for details.
    conceptbank_source = args.concept_bank.split("/")[-1].split(".")[0] 
    num_classes = len(classes)
    
    # Initialize the PCBM module.
    posthoc_layer = PosthocLinearMultilabelCBM(concept_bank, backbone_name=args.backbone_name, idx_to_class=idx_to_class, n_classes=num_classes)
    posthoc_layer = posthoc_layer.to(args.device)

    # We compute the projections and save to the output directory. This is to save time in tuning hparams / analyzing projections.
    train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls = load_or_compute_projections(args, backbone, posthoc_layer, train_loader, test_loader)
    
    run_info, weights, bias = run_linear_probe(args, (train_projs, train_lbls), (test_projs, test_lbls))
    
    # Convert from the SGDClassifier module to PCBM module.
    posthoc_layer.set_weights(weights=weights, bias=bias)

    # Sorry for the model path hack. Probably i'll change this later.
    model_path = os.path.join(args.out_dir,
                              f"pcbm_{args.dataset}__{args.backbone_name}__{conceptbank_source}__lam:{args.lam}__alpha:{args.alpha}__seed:{args.seed}.ckpt")
    torch.save(posthoc_layer, model_path)
    

    # # Analyze the 
    # concept_names = list(concept_bank.concept_names.copy())
    # analysis = analyze_classifier(posthoc_layer.residual_classifier, concept_names, k=5)

    # # Print and/or save the analysis results
    # print("Top-5 Concept Weights for Each Class:")
    # for class_idx, top_concepts in analysis.items():
    #     print(f"Class {idx_to_class[class_idx]}: {top_concepts}")


    # Again, a sad hack.. Open to suggestions
    run_info_file = model_path.replace("pcbm", "run_info-pcbm")
    run_info_file = run_info_file.replace(".ckpt", ".pkl")
    run_info_file = os.path.join(args.out_dir, run_info_file)
    
    with open(run_info_file, "wb") as f:
        pickle.dump(run_info, f)

    print(f"Model saved to : {model_path}")
    print(run_info)

if __name__ == "__main__":
    args = config()
    all_concepts = pickle.load(open(args.concept_bank, 'rb'))
    all_concept_names = list(all_concepts.keys())
    print(f"Bank path: {args.concept_bank}. {len(all_concept_names)} concepts will be used.")
    concept_bank = ConceptBank(all_concepts, args.device)

    # Get the backbone from the model zoo.
    backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
    backbone = backbone.to(args.device)
    backbone.eval()
    main(args, concept_bank, backbone, preprocess)