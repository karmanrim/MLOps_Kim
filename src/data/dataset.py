"""
Модуль для загрузки и предобработки данных
"""
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import logging

logger = logging.getLogger(__name__)


def get_data_transforms(augment: bool = False):
    """
    Получить трансформации для данных
    
    Args:
        augment: Использовать ли аугментацию для обучающей выборки
    
    Returns:
        train_transform, val_transform: Трансформации для обучения и валидации
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    return train_transform, val_transform


def load_fashion_mnist(data_dir: str, download: bool = True):
    """
    Загрузить Fashion-MNIST датасет
    
    Args:
        data_dir: Директория для сохранения данных
        download: Скачивать ли датасет, если его нет
    
    Returns:
        train_dataset, test_dataset: Обучающий и тестовый датасеты
    """
    logger.info(f"Загрузка Fashion-MNIST из {data_dir}")
    
    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transforms.ToTensor()
    )
    
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transforms.ToTensor()
    )
    
    logger.info(f"Загружено {len(train_dataset)} обучающих и {len(test_dataset)} тестовых образцов")
    
    return train_dataset, test_dataset


def create_dataloaders(
    train_dataset,
    test_dataset,
    batch_size: int = 64,
    num_workers: int = 4,
    train_split: float = 0.8,
    val_split: float = 0.1,
    random_seed: int = 42
):
    """
    Создать DataLoader'ы для обучения, валидации и тестирования
    
    Args:
        train_dataset: Обучающий датасет
        test_dataset: Тестовый датасет
        batch_size: Размер батча
        num_workers: Количество воркеров для загрузки данных
        train_split: Доля данных для обучения (остальное идет в валидацию)
        val_split: Не используется, оставлено для совместимости
        random_seed: Seed для воспроизводимости
    
    Returns:
        train_loader, val_loader, test_loader: DataLoader'ы
    """
    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    generator = torch.Generator().manual_seed(random_seed)
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=generator
    )
    
    logger.info(f"Разделение данных: train={len(train_subset)}, val={len(val_subset)}, test={len(test_dataset)}")
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def preprocess_data(config: dict):
    """
    Полная предобработка данных согласно конфигурации
    
    Args:
        config: Словарь с конфигурацией данных
    
    Returns:
        train_loader, val_loader, test_loader: DataLoader'ы
    """
    data_dir = config.get('data_dir', 'data')
    batch_size = config.get('batch_size', 64)
    num_workers = config.get('num_workers', 4)
    train_split = config.get('train_split', 0.8)
    random_seed = config.get('random_seed', 42)
    augment = config.get('augment', False)
    
    os.makedirs(data_dir, exist_ok=True)
    
    train_dataset, test_dataset = load_fashion_mnist(data_dir, download=True)
    
    train_transform, val_transform = get_data_transforms(augment=augment)
    
    train_dataset.transform = train_transform
    test_dataset.transform = val_transform
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        train_split=train_split,
        random_seed=random_seed
    )
    
    return train_loader, val_loader, test_loader

