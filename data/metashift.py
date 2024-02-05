from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch
import os

class MetashiftDataset(Dataset):
    def __init__(self, image_folder, metadata_file, transform=None):
        self.image_folder = image_folder
        self.data = pd.read_csv(metadata_file)
        self.transform = transform
        self.class_to_idx, self.idx_to_class = self._encode_labels()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, str(self.data.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name)
        # Use the encoded label
        label = self.class_to_idx[self.data.iloc[idx, 1]]

        if self.transform:
            image = self.transform(image)

        return image, label

    def _encode_labels(self):
        """
        Encodes string labels into integers and also prepares the reverse mapping.
        """
        # Get unique class labels and sort them alphabetically
        classes = self.data.iloc[:, 1].unique()
        classes.sort()
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
        return class_to_idx, idx_to_class

    def get_class_name(self, idx):
        """
        Get the class name for a given index.
        """
        return self.idx_to_class[idx]

def load_data(args, preprocess):
    # Create datasets
    train_dataset = MetashiftDataset(metadata_file='datasets/s1/s1_train_ft_metadata.csv',
                            image_folder='datasets/s1/s1_train_finetune',
                            transform=preprocess)

    test_dataset = MetashiftDataset(metadata_file='datasets/s1/s1_test_ft_metadata.csv',
                            image_folder='datasets/s1/s1_test_finetune',
                            transform=preprocess)
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Get idx_to_class from train dataset (assumed to be the same for test dataset)
    idx_to_class = train_dataset.idx_to_class

    return train_loader, test_loader, idx_to_class