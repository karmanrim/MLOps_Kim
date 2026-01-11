"""
Модуль для валидации данных
"""
import torch
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def validate_data_format(data: torch.Tensor, target: torch.Tensor) -> Tuple[bool, str]:
    """
    Проверка формата и структуры данных
    
    Args:
        data: Тензор с данными
        target: Тензор с метками
    
    Returns:
        (is_valid, error_message): Кортеж с результатом проверки
    """
    if not isinstance(data, torch.Tensor):
        return False, f"Данные должны быть torch.Tensor, получен {type(data)}"
    
    if not isinstance(target, torch.Tensor):
        return False, f"Метки должны быть torch.Tensor, получен {type(target)}"
    
    if len(data.shape) != 4:
        return False, f"Данные должны иметь форма (batch, channels, height, width), получена форма {data.shape}"
    
    if len(target.shape) != 1:
        return False, f"Метки должны иметь форму (batch,), получена {target.shape}"
    
    if data.shape[0] != target.shape[0]:
        return False, f"размер батча данных ({data.shape[0]}) не совпадает с размером меток ({target.shape[0]})"
    
    return True, ""


def validate_data_types(data: torch.Tensor, target: torch.Tensor) -> Tuple[bool, str]:
    """
    Проверка типов данных и диапазонов значений
    
    Args:
        data: Тензор с данными
        target: Тензор с метками
    
    Returns:
        (is_valid, error_message): Кортеж с результатом проверки
    """
    if data.dtype != torch.float32:
        return False, f"Данные должны быть float32, получен {data.dtype}"
    
    if target.dtype != torch.int64:
        return False, f"Метки должны быть int64, получен {target.dtype}"
    
    if data.min() < -1.0 or data.max() > 1.0:
        return False, f"Данные должны быть нормализованы в диапазоне [-1, 1], получен [{data.min()}, {data.max()}]"
    
    if target.min() < 0:
        return False, f"Метки не должны быть отрицательными, минимальное значение: {target.min()}"
    
    return True, ""


def validate_data_features(data: torch.Tensor, expected_channels: int = 1, 
                          expected_size: int = 28) -> Tuple[bool, str]:
    """
    Проверка наличия необходимых признаков
    
    Args:
        data: Тензор с данными
        expected_channels: Ожидаемое количество каналов
        expected_size: Ожидаемый размер изображения
    
    Returns:
        (is_valid, error_message): Кортеж с результатом проверки
    """
    if data.shape[1] != expected_channels:
        return False, f"Ожидается {expected_channels} канал(ов), получено {data.shape[1]}"
    
    if data.shape[2] != expected_size or data.shape[3] != expected_size:
        return False, f"Ожидается размер {expected_size}x{expected_size}, получен {data.shape[2]}x{data.shape[3]}"
    
    return True, ""


def validate_target_labels(target: torch.Tensor, num_classes: int = 10) -> Tuple[bool, str]:
    """
    Проверка наличия целевых меток и их корректности
    
    Args:
        target: Тензор с метками
        num_classes: Количество классов
    
    Returns:
        (is_valid, error_message): Кортеж с результатом проверки
    """
    if len(target) == 0:
        return False, "Метки не могут быть пустыми"
    
    if target.max() >= num_classes:
        return False, f"Максимальное значение метки ({target.max()}) должно быть меньше {num_classes}"
    
    unique_labels = torch.unique(target)
    if len(unique_labels) == 0:
        return False, "Не найдено уникальных меток"
    
    return True, ""


def validate_dataset(dataloader, num_classes: int = 10, 
                    expected_channels: int = 1, expected_size: int = 28) -> Tuple[bool, List[str]]:
    """
    Полная валидация датасета
    
    Args:
        dataloader: DataLoader для проверки
        num_classes: Количество классов
        expected_channels: Ожидаемое количество каналов
        expected_size: Ожидаемый размер изображения
    
    Returns:
        (is_valid, errors): Кортеж с результатом проверки и списком ошибок
    """
    errors = []
    
    for batch_idx, (data, target) in enumerate(dataloader):
        is_valid, error = validate_data_format(data, target)
        if not is_valid:
            errors.append(f"Batch {batch_idx}: {error}")
            continue
        
        is_valid, error = validate_data_types(data, target)
        if not is_valid:
            errors.append(f"Batch {batch_idx}: {error}")
            continue
        
        is_valid, error = validate_data_features(data, expected_channels, expected_size)
        if not is_valid:
            errors.append(f"Batch {batch_idx}: {error}")
            continue
        
        is_valid, error = validate_target_labels(target, num_classes)
        if not is_valid:
            errors.append(f"Batch {batch_idx}: {error}")
            continue
        
        if batch_idx >= 10:
            break
    
    return len(errors) == 0, errors

