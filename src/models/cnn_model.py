"""
Архитектура CNN модели для классификации изображений Fashion-MNIST
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
import json
import os
from typing import Optional


class CNNConfig(PretrainedConfig):
    """Конфигурация для CNN модели"""
    model_type = "cnn"
    
    def __init__(
        self,
        num_classes: int = 10,
        input_size: int = 28,
        hidden_dims: list = None,
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 64]
        self.dropout = dropout


class CNNModel(PreTrainedModel):
    """
    CNN модель для классификации изображений Fashion-MNIST
    Наследуется от PreTrainedModel для совместимости с Hugging Face
    """
    config_class = CNNConfig
    
    def __init__(self, config: CNNConfig):
        super().__init__(config)
        self.config = config
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        fc_input_size = (config.input_size // 8) * (config.input_size // 8) * 128
        
        self.fc1 = nn.Linear(fc_input_size, config.hidden_dims[0])
        self.fc2 = nn.Linear(config.hidden_dims[0], config.hidden_dims[1])
        self.fc3 = nn.Linear(config.hidden_dims[1], config.num_classes)
        
        self.dropout = nn.Dropout(config.dropout)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Входной тензор формы (batch_size, 1, 28, 28)
        
        Returns:
            Логиты для каждого класса
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


def create_model_from_config(model_config: dict) -> CNNModel:
    """
    Создать модель из конфигурации
    
    Args:
        model_config: Словарь с конфигурацией модели
    
    Returns:
        Инициализированная модель
    """
    config = CNNConfig(
        num_classes=model_config.get('num_classes', 10),
        input_size=model_config.get('input_size', 28),
        hidden_dims=model_config.get('hidden_dims', [128, 64]),
        dropout=model_config.get('dropout', 0.5)
    )
    
    model = CNNModel(config)
    return model

