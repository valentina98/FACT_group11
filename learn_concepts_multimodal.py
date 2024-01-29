import requests
import os
import pickle
import torch
import clip
import argparse
import numpy as np
from tqdm import tqdm
import re
import sklearn
from sklearn.datasets import fetch_20newsgroups
import torch.nn as nn

class DimensionReducer(nn.Module):
    def __init__(self, input_dim=1024, output_dim=768):
        super(DimensionReducer, self).__init__()
        self.reducer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.to(dtype=self.reducer.weight.dtype)
        return self.reducer(x)

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--classes", default="cifar10", type=str)
    parser.add_argument("--backbone-name", default="clip:RN50", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--recurse", default=1, type=int, help="How many times to recurse on the conceptnet graph")
    return parser.parse_args()


def get_single_concept_data(cls_name,type="image"):
    if cls_name in concept_cache:
        return concept_cache[cls_name]
    
    all_concepts = []
    cls_name = cls_name.lower()
    if type == "nlp":
        
        # RelatedTo relations
        related_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/RelatedTo&start=/c/en/{}"
        obj = requests.get(related_query.format(cls_name, cls_name)).json()
        for edge in obj["edges"]:
            all_concepts.append(edge['end']['label'])

        # Synonym relations
        synonym_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/Synonym&start=/c/en/{}"
        obj = requests.get(synonym_query.format(cls_name, cls_name)).json()
        for edge in obj["edges"]:
            all_concepts.append(edge['end']['label'])

        # UsedFor relations
        usedfor_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/UsedFor&start=/c/en/{}"
        obj = requests.get(usedfor_query.format(cls_name, cls_name)).json()
        for edge in obj["edges"]:
            all_concepts.append(edge['end']['label'])

        formof_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/FormOf&start=/c/en/{}"
        obj = requests.get(formof_query.format(cls_name, cls_name)).json()
        for edge in obj["edges"]:
            all_concepts.append(edge['end']['label'])

        isa_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/IsA&start=/c/en/{}"
        obj = requests.get(isa_query.format(cls_name, cls_name)).json()
        for edge in obj["edges"]:
            all_concepts.append(edge['end']['label'])

        atlocation_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/AtLocation&start=/c/en/{}"
        obj = requests.get(atlocation_query.format(cls_name, cls_name)).json()
        for edge in obj["edges"]:
            all_concepts.append(edge['end']['label'])
        
    else:
        # Has relations
        has_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/HasA&start=/c/en/{}"
        obj = requests.get(has_query.format(cls_name, cls_name)).json()
        for edge in obj["edges"]:
            all_concepts.append(edge['end']['label'])
        
        # Made of relations
        madeof_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/MadeOf&start=/c/en/{}"
        obj = requests.get(madeof_query.format(cls_name, cls_name)).json()
        for edge in obj["edges"]:
            all_concepts.append(edge['end']['label'])
        
        # Properties of things
        property_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/HasProperty&start=/c/en/{}"
        obj = requests.get(property_query.format(cls_name, cls_name)).json()
        for edge in obj["edges"]:
            all_concepts.append(edge['end']['label'])
        
        # Categorization concepts
        is_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/IsA&start=/c/en/{}"
        obj = requests.get(is_query.format(cls_name, cls_name)).json()
        for edge in obj["edges"]:
            if edge["weight"] <= 1:
                continue
            all_concepts.append(edge['end']['label'])
        
        # Parts of things
        parts_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/PartOf&end=/c/en/{}"
        obj = requests.get(parts_query.format(cls_name, cls_name)).json()
        for edge in obj["edges"]:
            all_concepts.append(edge['start']['label'])
    
    all_concepts = [c.lower() for c in all_concepts]
    # Drop the "a " for concepts defined like "a {concept}".
    all_concepts = [c.replace("a ", "") for c in all_concepts]
    # Drop all empty concepts.
    all_concepts = [c for c in all_concepts if c!=""]
    # Make each concept unique in the set.
    all_concepts = set(all_concepts)
    
    concept_cache[cls_name] = all_concepts
    
    return all_concepts

def preprocess_text_for_CLIP(text, max_length=77):
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", '', text)

    tokens = clip.tokenize([text], truncate=True)[0].numpy()

    max_length -= 2

    truncated_tokens = tokens[:max_length + 1]

    processed_text = clip._tokenizer.decode(truncated_tokens.tolist())

    return processed_text


def get_concept_data(all_classes,type="image"):
    all_concepts = set()
    # Collect concepts that are relevant to each class
    for cls_name in all_classes:
        print(f"Pulling concepts for {cls_name}")
        all_concepts |= get_single_concept_data(cls_name,type=type)
    return all_concepts


def clean_concepts(scenario_concepts):
    """
    Clean the plurals, trailing whitespaces etc.
    """
    from nltk.stem.wordnet import WordNetLemmatizer
    import nltk

    # We use nltk to handle plurals, multiples of the same words etc.
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    Lem = WordNetLemmatizer()

    scenario_concepts_rec = []
    for c_prev in scenario_concepts:
        c = c_prev
        c = c.strip()
        c_subwords = c.split(" ")
        # If a concept is made of more than 2 words, we drop it.
        if len(c_subwords) > 2:
            print("skipping long concept", c_prev)
            continue
        # Lemmatize words to help eliminate non-unique concepts etc.
        for i, csw in enumerate(c_subwords):
            c_subwords[i] = Lem.lemmatize(csw)
        lemword = " ".join(c_subwords)
        if c_prev == lemword:
            scenario_concepts_rec.append(c)
        else:
            if lemword in scenario_concepts:
                print(c, lemword)
            else:
                scenario_concepts_rec.append(c)
    scenario_concepts_rec = list(set(scenario_concepts_rec))
    return scenario_concepts_rec

@torch.no_grad()
def learn_conceptbank(args, concept_list, scenario):
    concept_dict = {}
    for concept in tqdm(concept_list):
        # Note: You can try other forms of prompting, e.g. "photo of {concept}" etc. here.
        text = clip.tokenize(f"{concept}").to("cuda")
        text_features = model.encode_text(text).cpu().numpy()
        text_features = text_features/np.linalg.norm(text_features)
        if "20ng" in args.classes:
            dimension_reducer = DimensionReducer(input_dim=1024, output_dim=768).to(args.device)
            dimension_reducer.eval()
            text_features = dimension_reducer(torch.from_numpy(text_features).to(args.device)).cpu().numpy()
        # store concept vectors in a dictionary. Adding the additional terms to be consistent with the
        # `ConceptBank` class (see `concepts/concept_utils.py`).
        concept_dict[concept] = (text_features, None, None, 0, {})

    print(f"# concepts: {len(concept_dict)}")
    concept_dict_path = os.path.join(args.out_dir, f"multimodal_concept_{args.backbone_name}_{scenario}_recurse:{args.recurse}.pkl")
    pickle.dump(concept_dict, open(concept_dict_path, 'wb'))
    print(f"Dumped to : {concept_dict_path}")

if __name__ == "__main__":
    args = config()
    model, _ = clip.load(args.backbone_name.split(":")[1], device=args.device, download_root=args.out_dir)
    concept_cache = {}
    
    if args.classes == "cifar10":
        # Pull CIFAR10 to get the class names.
        from torchvision import datasets
        cifar10_ds = datasets.CIFAR10(root=args.out_dir, train=True, download=True)
        # Get the class names.
        all_classes = list(cifar10_ds.classes)
        # Get the names of all concepts.
        all_concepts = get_concept_data(all_classes)
        # Clean the concepts for uniques, plurals etc. 
        all_concepts = clean_concepts(all_concepts)     
        all_concepts = list(set(all_concepts).difference(set(all_classes)))
        # If we'd like to recurse in the conceptnet graph, specify `recurse > 1`.
        for i in range(1, args.recurse):
            all_concepts = get_concept_data(all_concepts)
            all_concepts = list(set(all_concepts))
            all_concepts = clean_concepts(all_concepts)
            all_concepts = list(set(all_concepts).difference(set(all_classes)))
        # Generate the concept bank.
        learn_conceptbank(args, all_concepts, args.classes)
        
    elif args.classes == "cifar100":
        from torchvision import datasets
        cifar100_ds = datasets.CIFAR100(root=args.out_dir, train=True, download=True)
        all_classes = list(cifar100_ds.classes)
        all_concepts = get_concept_data(all_classes)
        all_concepts = clean_concepts(all_concepts)
        all_concepts = list(set(all_concepts).difference(set(all_classes)))
        # If we'd like to recurse in the conceptnet graph, specify `recurse > 1`.
        for i in range(1, args.recurse):
            all_concepts = get_concept_data(all_concepts)
            all_concepts = list(set(all_concepts))
            all_concepts = clean_concepts(all_concepts)
            all_concepts = list(set(all_concepts).difference(set(all_classes)))
        learn_conceptbank(args, all_concepts, args.classes)

    elif args.classes == "20ng":
        newsgroup_name_mapping = {
            "alt.atheism": "Atheism",
            "comp.graphics": "Computer Graphics",
            "comp.os.ms-windows.misc": "Windows Operating System",
            "comp.sys.ibm.pc.hardware": "IBM",
            "comp.sys.mac.hardware": "Macintosh",
            "comp.windows.x": "GUI",
            "misc.forsale": "Marketplace",
            "rec.autos": "Automobiles",
            "rec.motorcycles": "Motorbikes",
            "rec.sport.baseball": "Baseball",
            "rec.sport.hockey": "Hockey",
            "sci.crypt": "Cryptography",
            "sci.electronics": "Electronics",
            "sci.med": "Medicine",
            "sci.space": "Space Exploration",
            "soc.religion.christian": "Christianity",
            "talk.politics.guns": "Gun Politics",
            "talk.politics.mideast": "Middle Eastern Politics",
            "talk.politics.misc": "General Politics",
            "talk.religion.misc": "Religion"
        }
        newsgroups_data = fetch_20newsgroups(subset='all')
        all_classes = [newsgroup_name_mapping.get(name, name) for name in newsgroups_data.target_names]

        all_concepts = get_concept_data(all_classes,type="nlp")
        print(all_concepts)
        all_concepts = clean_concepts(all_concepts)
        all_concepts = list(set(all_concepts).difference(set(all_classes)))

        for i in range(1, args.recurse):
            all_concepts = get_concept_data(all_concepts,type="nlp")
            all_concepts = list(set(all_concepts))
            all_concepts = clean_concepts(all_concepts)
            all_concepts = list(set(all_concepts).difference(set(all_classes)))

        learn_conceptbank(args, all_concepts, args.classes)

    else:
        raise ValueError(f"Unknown classes: {args.classes}. Define your dataset here!")
