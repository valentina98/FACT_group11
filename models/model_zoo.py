import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision import transforms


class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ResNetTop(nn.Module):
    def __init__(self, original_model):
        super(ResNetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x

# MobileNet
class MobileNetBottom(nn.Module):
    def __init__(self, original_model):
        super(MobileNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

class MobileNetTop(nn.Module):
    def __init__(self, original_model):
        super(MobileNetTop, self).__init__()
        self.classifier = nn.Sequential(*[list(original_model.children())[-1]])

    def forward(self, x):
        x = self.classifier(x)
        x = nn.Softmax(dim=-1)(x)
        return x

def get_model(args, backbone_name="resnet18_cub", full_model=False):
    if "clip" in backbone_name:
        import clip
        # We assume clip models are passed of the form : clip:RN50
        clip_backbone_name = backbone_name.split(":")[1]
        backbone, preprocess = clip.load(clip_backbone_name, device=args.device, download_root=args.out_dir)
        backbone = backbone.eval()
        model = None
    
    elif backbone_name == "resnet18_cub":
        from pytorchcv.model_provider import get_model as ptcv_get_model
        model = ptcv_get_model(backbone_name, pretrained=True, root=args.out_dir)
        backbone, model_top = ResNetBottom(model), ResNetTop(model)
        cub_mean_pxs = np.array([0.5, 0.5, 0.5])
        cub_std_pxs = np.array([2., 2., 2.])
        preprocess = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(cub_mean_pxs, cub_std_pxs)
            ])
    
    elif backbone_name.lower() == "ham10000_inception":
        from .derma_models import get_derma_model
        model, backbone, model_top = get_derma_model(args, backbone_name.lower())
        preprocess = transforms.Compose([
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                      ])

    # New: Resnet50 for HAM10000
    elif backbone_name.lower() == "ham10000_resnet50":
        from .derma_models import get_derma_model
        model, backbone, model_top = get_derma_model(args, backbone_name.lower())
        preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                      ])
    
    # New: Densenet for HAM10000
    elif backbone_name.lower() == "ham10000_densenet":
        from .derma_models import get_derma_model
        model, backbone, model_top = get_derma_model(args, backbone_name.lower())
        preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                      ])
        
    # New: mobilenet_w1_cub and proxylessnas_mobile_cub for CUB
    elif backbone_name.lower() == "mobilenet_w1_cub" or backbone_name.lower() == "proxylessnas_mobile_cub":
        from pytorchcv.model_provider import get_model as ptcv_get_model
        model = ptcv_get_model(backbone_name, pretrained=True, root=args.out_dir)
        backbone, model_top = MobileNetBottom(model), MobileNetTop(model)
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    elif backbone_name.lower() == "bert":
        model_name = 'bert-base-uncased'
        backbone = BertModel.from_pretrained(model_name)
        preprocess = None
             
    elif backbone_name == "resnet18":
        from pytorchcv.model_provider import get_model as ptcv_get_model
        model = ptcv_get_model(backbone_name, pretrained=True, root=args.out_dir)
        backbone, model_top = ResNetBottom(model), ResNetTop(model)
        imagenet_mean_pxs = np.array([0.485, 0.456, 0.406])
        imagenet_std_pxs = np.array([0.229, 0.224, 0.225])
        preprocess = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean_pxs, imagenet_std_pxs)
            ]) 
    
    else:
        raise ValueError(backbone_name)

    if full_model:
        return model, backbone, preprocess
    else:
        return backbone, preprocess


