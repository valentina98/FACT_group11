from torchvision import datasets
import torch
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
import numpy as np 
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def get_dataset(args, preprocess=None):
    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root=args.out_dir, train=True,
                                    download=True, transform=preprocess)
        testset = datasets.CIFAR10(root=args.out_dir, train=False,
                                    download=True, transform=preprocess)
        classes = trainset.classes
        class_to_idx = {c: i for (i,c) in enumerate(classes)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers)
    
    
    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root=args.out_dir, train=True,
                                    download=True, transform=preprocess)
        testset = datasets.CIFAR100(root=args.out_dir, train=False,
                                    download=True, transform=preprocess)
        classes = trainset.classes
        class_to_idx = {c: i for (i,c) in enumerate(classes)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers)
    
    elif args.dataset == "cifar10_val":

        dataset = datasets.CIFAR10(root=args.out_dir, train=True,
                                download=True, transform=preprocess)
        testset = datasets.CIFAR10(root=args.out_dir, train=False,
                                download=True, transform=preprocess)
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(0.2 * num_train))

        np.random.shuffle(indices)

        train_idx, val_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                sampler=train_sampler, num_workers=args.num_workers)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                sampler=val_sampler, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers)

        classes = dataset.classes
        
        return train_loader, test_loader, classes, val_loader

    elif args.dataset == "cifar100_val":
        dataset = datasets.CIFAR100(root=args.out_dir, train=True,
                                    download=True, transform=preprocess)
        testset = datasets.CIFAR100(root=args.out_dir, train=False,
                                    download=True, transform=preprocess)
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(0.2 * num_train))

        np.random.shuffle(indices)

        train_idx, val_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                sampler=train_sampler, num_workers=args.num_workers)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                sampler=val_sampler, num_workers=args.num_workers)
        
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers)

        classes = dataset.classes

        return train_loader, test_loader, classes, val_loader

    elif args.dataset == "cub":
        from .cub import load_cub_data
        from .constants import CUB_PROCESSED_DIR, CUB_DATA_DIR
        from torchvision import transforms
        num_classes = 200
        TRAIN_PKL = os.path.join(CUB_PROCESSED_DIR, "train.pkl")
        TEST_PKL = os.path.join(CUB_PROCESSED_DIR, "test.pkl")
        normalizer = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        train_loader = load_cub_data([TRAIN_PKL], use_attr=False, no_img=False, 
            batch_size=args.batch_size, uncertain_label=False, image_dir=CUB_DATA_DIR, resol=224, normalizer=normalizer,
            n_classes=num_classes, resampling=True)

        test_loader = load_cub_data([TEST_PKL], use_attr=False, no_img=False, 
                batch_size=args.batch_size, uncertain_label=False, image_dir=CUB_DATA_DIR, resol=224, normalizer=normalizer,
                n_classes=num_classes, resampling=True)

        classes = open(os.path.join(CUB_DATA_DIR, "classes.txt")).readlines()
        classes = [a.split(".")[1].strip() for a in classes]
        idx_to_class = {i: classes[i] for i in range(num_classes)}
        classes = [classes[i] for i in range(num_classes)]
        print(len(classes), "num classes for cub")
        print(len(train_loader.dataset), "training set size")
        print(len(test_loader.dataset), "test set size")
        

    elif args.dataset == "ham10000":
        from .derma_data import load_ham_data
        train_loader, test_loader, idx_to_class = load_ham_data(args, preprocess)
        class_to_idx = {v:k for k,v in idx_to_class.items()}
        classes = list(class_to_idx.keys())
    
    elif args.dataset == "20ng":
        train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(train_data.target)
        y_test = encoder.transform(test_data.target)

        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TextDataset(train_data.data, y_train_tensor)
        test_dataset = TextDataset(test_data.data, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        idx_to_class = {i: class_name for i, class_name in enumerate(train_data.target_names)}
        classes = train_data.target_names

    elif args.dataset == "20ng_val":

        train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(train_data.target)
        y_test = encoder.transform(test_data.target)

        X_train, X_val, y_train, y_val = train_test_split(
            train_data.data, y_train, test_size=0.2, random_state=42)

        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        train_dataset = TextDataset(X_train, y_train_tensor)
        val_dataset = TextDataset(X_val, y_val_tensor)
        test_dataset = TextDataset(test_data.data, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        idx_to_class = {i: class_name for i, class_name in enumerate(train_data.target_names)}
        classes = train_data.target_names
        return train_loader, test_loader, classes, val_loader


    else:
        raise ValueError(args.dataset)

    return train_loader, test_loader, idx_to_class, classes

