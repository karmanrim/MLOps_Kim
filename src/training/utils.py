"""
Утилиты для обучения модели
"""
import torch
import yaml
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Загрузка конфигурации из YAML файла
    
    Args:
        config_path: Путь к конфигурационному файлу
    
    Returns:
        Словарь с конфигурацией
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def set_random_seed(seed: int):
    """
    Установка random seed для воспроизводимости
    
    Args:
        seed: Значение seed
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Получение устройства для обучения (CPU или GPU)
    
    Returns:
        Устройство
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Создание оптимизатора из конфигурации
    
    Args:
        model: Модель для оптимизации
        config: Конфигурация обучения
    
    Returns:
        Оптимизатор
    """
    optimizer_name = config.get('optimizer', 'Adam').lower()
    learning_rate = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)
    
    if optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Неизвестный оптимизатор: {optimizer_name}")


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
    """
    Создание scheduler из конфигурации
    
    Args:
        optimizer: Оптимизатор
        config: Конфигурация обучения
    
    Returns:
        Scheduler или None
    """
    scheduler_type = config.get('scheduler')
    
    if scheduler_type == 'StepLR':
        step_size = config.get('step_size', 10)
        gamma = config.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    return None

