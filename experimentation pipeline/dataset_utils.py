import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split


def get_data_transforms(config):
    """Get data transforms based on config"""
    data_config = config['data']
    
    # Base transforms
    train_transforms = []
    test_transforms = [transforms.ToTensor()]

    # g = torch.Generator()
    # g.manual_seed(data_config.get('transforms_seed', 42))

    # Add augmentations for training
    if data_config['augmentation']['random_crop']:
        train_transforms.append(transforms.RandomCrop(32, padding=4))

    if data_config['augmentation']['random_horizontal_flip']:
        train_transforms.append(transforms.RandomHorizontalFlip())

    if data_config['augmentation']['rand_augment']:
        train_transforms.append(transforms.RandAugment(
            num_ops=data_config['augmentation']['rand_augment_num_ops'],
            magnitude=data_config['augmentation']['rand_augment_magnitude']
        ))
    
    train_transforms.append(transforms.ToTensor())

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
    
    val_split = data_config.get('val_split', 0.1)
    generator = torch.Generator().manual_seed(data_config.get('split_seed', 42))
    num_train = len(train_dataset)
    num_val = int(num_train * val_split)
    num_train = num_train - num_val

    train_subset, val_subset = random_split(train_dataset, [num_train, num_val], generator= generator)

    train_dataset = train_subset
    val_dataset = val_subset
    val_dataset.transform = test_dataset.transform  # Use same transform as test dataset

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True,
        worker_init_fn=lambda x: torch.manual_seed(data_config.get('worker_init_seed', 42) + x)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=True,
        worker_init_fn=lambda x: torch.manual_seed(data_config.get('worker_init_seed', 42) + x)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True,
        worker_init_fn=lambda x: torch.manual_seed(data_config.get('worker_init_seed', 42) + x)
    )
    
    return train_loader, test_loader, val_loader