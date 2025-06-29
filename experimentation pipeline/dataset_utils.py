import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_transforms(config):
    """Get data transforms based on config"""
    data_config = config['data']
    
    # Base transforms
    train_transforms = [transforms.ToTensor()]
    test_transforms = [transforms.ToTensor()]
    
    # Add augmentations for training
    if data_config['augmentation']['random_crop']:
        train_transforms.insert(0, transforms.RandomCrop(32, padding=4))
    
    if data_config['augmentation']['random_horizontal_flip']:
        train_transforms.insert(-1, transforms.RandomHorizontalFlip())
    
    # Add normalization
    if data_config['augmentation']['normalize']:
        normalize = transforms.Normalize(
            mean=data_config['normalize_mean'],
            std=data_config['normalize_std']
        )
        train_transforms.append(normalize)
        test_transforms.append(normalize)
    
    transform_train = transforms.Compose(train_transforms)
    transform_test = transforms.Compose(test_transforms)
    
    return transform_train, transform_test


def get_datasets(config):
    """Get datasets based on config"""
    data_config = config['data']
    transform_train, transform_test = get_data_transforms(config)
    
    if data_config['dataset'].lower() == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform_test
        )
    else:
        raise ValueError(f"Unsupported dataset: {data_config['dataset']}")
    
    return train_dataset, test_dataset


def get_data_loaders(config):
    """Get data loaders based on config"""
    data_config = config['data']
    train_dataset, test_dataset = get_datasets(config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, test_loader