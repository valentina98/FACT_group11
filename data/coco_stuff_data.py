from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

class CocoStuffDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        while True:
            try:
                image = Image.open(self.filepaths[idx]).convert('RGB')
                label = self.labels[idx]

                if self.transform:
                    image = self.transform(image)

                # print(f"Label type: {type(label)}, Label value: {label}")

                return image, label
            except Exception as e:
                print(f"Error loading image: {self.filepaths[idx]}, Error: {e}")
                # Optionally, remove the invalid sample from the dataset
                del self.filepaths[idx]
                del self.labels[idx]

                # If the dataset becomes empty, raise an exception or return a default value
                if not self.filepaths:
                    raise ValueError("All images in the dataset are invalid")

def sample_or_upsample(dataset_view, num_samples, seed):
    sampled_filepaths = [sample.filepath for sample in dataset_view.take(num_samples, seed)]
    
    # Upsample if there are not enough images
    while len(sampled_filepaths) < num_samples:
        additional_sample = dataset_view.shuffle(seed).take(1).first()
        sampled_filepaths.append(additional_sample.filepath)
    
    return sampled_filepaths

def load_coco_stuff_data(args, biased_classes, num_train_samples_per_class=500, num_val_samples_per_class=250):
    """
    Loads the COCO dataset with a focus on specified biased classes, applies preprocessing,
    and returns PyTorch DataLoaders for training and validation sets.

    Args:
    - args: Arguments containing dataset configurations like batch size and seed.
    - biased_classes (List): List of biased classes to focus on.
    - num_test_samples_per_class (int): Number of training samples per class.
    - num_val_samples_per_class (int): Number of validation samples per class.

    Returns:
    - train_loader (DataLoader): DataLoader for the training set.
    - val_loader (DataLoader): DataLoader for the validation set.
    - idx_to_class (Dict): Mapping from class index to class name.
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = foz.load_zoo_dataset("coco-2017", split="train", label_types=["detections"], classes=biased_classes, max_samples=num_train_samples_per_class*len(biased_classes))
    val_dataset = foz.load_zoo_dataset("coco-2017", split="validation", label_types=["detections"], classes=biased_classes, max_samples=num_val_samples_per_class*len(biased_classes))

    train_filepaths, train_labels = [], []
    val_filepaths, val_labels = [], []
    for class_name in biased_classes:
        
        # Create a filter expression for each class
        filter_expr = F("label") == class_name

        # Apply the filter to the dataset views
        train_view = train_dataset.filter_labels("ground_truth", filter_expr)
        val_view = val_dataset.filter_labels("ground_truth", filter_expr)
        
        sampled_train_filepaths = sample_or_upsample(train_view, num_train_samples_per_class, args.seed)
        sampled_val_filepaths = sample_or_upsample(val_view, num_val_samples_per_class, args.seed)

        # Print the number of samples in each view after filtering
        print(f"Class '{class_name}': Train View Size = {train_view.count()}, Validation View Size = {val_view.count()}, Sampled Train View Size = {len(sampled_train_filepaths)}, Sampled Validation View Size = {len(sampled_val_filepaths)}")

        train_filepaths.extend(sampled_train_filepaths)
        train_labels.extend([class_name] * len(sampled_train_filepaths))

        val_filepaths.extend(sampled_val_filepaths)
        val_labels.extend([class_name] * len(sampled_val_filepaths))

    train_data = CocoStuffDataset(train_filepaths, train_labels, transform=preprocess)
    val_data = CocoStuffDataset(val_filepaths, val_labels, transform=preprocess)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    idx_to_class = {i: class_name for i, class_name in enumerate(biased_classes)}

    # debug
    for batch_data, batch_labels in train_loader:
        print(type(batch_data), type(batch_labels), len(batch_labels), type(batch_labels[0]), batch_labels[0])
        break 
            
    return train_loader, val_loader, idx_to_class
