from torchvision import datasets
import torch
import os

def sample_or_upsample(dataset_view, num_samples):
    """
    Samples or upsamples a FiftyOne dataset view to have the specified number of samples.

    Args:
    - dataset_view (fo.View): The FiftyOne dataset view to sample from.
    - num_samples (int): The desired number of samples.

    Returns:
    - List: A list of sampled filepaths.
    """
    # Your logic to sample or upsample the dataset view
    # This function should return a list of filepaths for the sampled images
    # ...

def create_pytorch_dataset(filepaths, labels, transform):
    """
    Creates a PyTorch dataset from filepaths and labels.

    Args:
    - filepaths (List): List of image filepaths.
    - labels (List): Corresponding labels for the images.
    - transform (callable): Transform to be applied on a sample.

    Returns:
    - torch.utils.data.Dataset
    """
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, filepaths, labels, transform=None):
            self.filepaths = filepaths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.filepaths)

        def __getitem__(self, idx):
            image = Image.open(self.filepaths[idx])
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label

    return CustomDataset(filepaths, labels, transform)


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

    elif args.dataset == "coco_stuff":
        # The list of biased classes provided in the paper
        classes = ["cup", "handbag", "apple", "car", "bus", "potted plant",
                   "spoon", "microwave", "keyboard", "clock", "hair drier", "skateboard"]
        from .coco_stuff_data import load_coco_stuff_data
        train_loader, test_loader, idx_to_class = load_coco_stuff_data(args, classes, 500, 250)

        ########### TODO remove
        # Plot
        import matplotlib.pyplot as plt

        def show_sample_images(data_loader, idx_to_class, num_images=5):
            images, labels = next(iter(data_loader))
            fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
            for i, (image, label) in enumerate(zip(images[:10], labels[:10])):
                axes[i].imshow(image.permute(1, 2, 0))
                axes[i].set_title(idx_to_class[label])
                axes[i].axis('off')
            plt.show()

        print("Displaying Training Samples:")
        show_sample_images(train_loader, idx_to_class)

        print("Displaying Validation Samples:")
        show_sample_images(test_loader, idx_to_class)
        ##########
                
    else:
        raise ValueError(args.dataset)

    return train_loader, test_loader, idx_to_class, classes

