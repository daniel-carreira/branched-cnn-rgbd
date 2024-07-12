import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class CustomDataset(Dataset):
    def __init__(self, rgb_folder, depth_folder, transform=None, normalized=False):
        self.rgb_dataset = datasets.ImageFolder(root=rgb_folder, transform=transform)
        depth_transforms = transforms.Compose([transforms.Grayscale(), transform])
        self.depth_dataset = datasets.ImageFolder(root=depth_folder, transform=depth_transforms)
        
        if normalized:
            # RGB
            rgb_aux = [img[0] for img in self.rgb_dataset]
            rgb_aux = torch.stack(rgb_aux)
            rgb_mean = (rgb_aux.data.float() / 255.0).mean().item()
            rgb_std = (rgb_aux.data.float() / 255.0).std().item()
            
            rgb_transforms = transforms.Compose([transform, transforms.Normalize(rgb_mean, rgb_std)])
            self.rgb_dataset = datasets.ImageFolder(root=rgb_folder, transform=rgb_transforms)
            
            # Depth
            depth_aux = [img[0] for img in self.depth_dataset]
            depth_aux = torch.stack(depth_aux)
            depth_mean = (depth_aux.data.float() / 255.0).mean().item()
            depth_std = (depth_aux.data.float() / 255.0).std().item()
            
            depth_transforms = transforms.Compose([depth_transforms, transforms.Normalize(depth_mean, depth_std)])
            self.depth_dataset = datasets.ImageFolder(root=depth_folder, transform=depth_transforms)

    def __len__(self):
        return len(self.rgb_dataset)
    
    def __getitem__(self, index):
        rgb, label = self.rgb_dataset[index]
        depth, _ = self.depth_dataset[index]
        return rgb, depth, label