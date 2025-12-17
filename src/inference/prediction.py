"""
Модуль для конвертации предсказаний модели в финальные значения
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def convert_logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """
    Конвертация логитов в вероятности классов
    
    Args:
        logits: Сырые выходы модели (логиты)
    
    Returns:
        Вероятности классов
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits должен быть torch.Tensor, получен {type(logits)}")
    
    if len(logits.shape) != 2:
        raise ValueError(f"logits должен иметь форму (batch_size, num_classes), получена {logits.shape}")
    
    probabilities = F.softmax(logits, dim=1)
    return probabilities


def convert_probabilities_to_class(probabilities: torch.Tensor) -> torch.Tensor:
    """
    Конвертация вероятностей в итоговый класс
    
    Args:
        probabilities: Вероятности классов
    
    Returns:
        Индексы классов с максимальной вероятностью
    """
    if not isinstance(probabilities, torch.Tensor):
        raise TypeError(f"probabilities должен быть torch.Tensor, получен {type(probabilities)}")
    
    if len(probabilities.shape) != 2:
        raise ValueError(f"probabilities должен иметь форму (batch_size, num_classes), получена {probabilities.shape}")
    
    predicted_classes = torch.argmax(probabilities, dim=1)
    return predicted_classes


def convert_logits_to_class(logits: torch.Tensor) -> torch.Tensor:
    """
    Прямая конвертация логитов в класс (без промежуточных вероятностей)
    
    Args:
        logits: Сырые выходы модели (логиты)
    
    Returns:
        Индексы классов с максимальной вероятностью
    """
    probabilities = convert_logits_to_probabilities(logits)
    classes = convert_probabilities_to_class(probabilities)
    return classes


def convert_to_numpy(predictions: torch.Tensor) -> np.ndarray:
    """
    Конвертация предсказаний в numpy array для API
    
    Args:
        predictions: Предсказания в виде torch.Tensor
    
    Returns:
        Предсказания в виде numpy array
    """
    if isinstance(predictions, torch.Tensor):
        return predictions.detach().cpu().numpy()
    elif isinstance(predictions, np.ndarray):
        return predictions
    else:
        raise TypeError(f"predictions должен быть torch.Tensor или numpy.ndarray, получен {type(predictions)}")


def process_model_output(logits: torch.Tensor, return_probabilities: bool = False) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
    """
    Полная обработка выходов модели для API
    
    Args:
        logits: Сырые выходы модели
        return_probabilities: Возвращать ли вероятности классов
    
    Returns:
        Словарь с обработанными предсказаниями
    """
    probabilities = convert_logits_to_probabilities(logits)
    predicted_classes = convert_probabilities_to_class(probabilities)
    
    result = {
        'predicted_class': convert_to_numpy(predicted_classes)
    }
    
    if return_probabilities:
        result['probabilities'] = convert_to_numpy(probabilities)
    
    return result


def validate_prediction_output(logits: torch.Tensor) -> Tuple[bool, str]:
    """
    Валидация выходов модели
    
    Args:
        logits: Сырые выходы модели
    
    Returns:
        (is_valid, error_message): Кортеж с результатом проверки
    """
    if not isinstance(logits, torch.Tensor):
        return False, f"logits должен быть torch.Tensor, получен {type(logits)}"
    
    if len(logits.shape) != 2:
        return False, f"logits должен иметь форму (batch_size, num_classes), получена {logits.shape}"
    
    if logits.shape[0] == 0:
        return False, "logits не может быть пустым"
    
    if not torch.isfinite(logits).all():
        return False, "logits содержит нефинитные значения (NaN или Inf)"
    
    return True, ""

