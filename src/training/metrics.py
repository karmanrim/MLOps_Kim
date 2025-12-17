"""
Модуль для расчета метрик качества модели
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred):
    """
    Рассчитать метрики качества модели
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
    
    Returns:
        Словарь с метриками
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def validate_model(model, test_loader, device, num_classes=10):
    """
    Полная валидация модели на тестовой выборке
    
    Args:
        model: Модель для валидации
        test_loader: DataLoader для тестовых данных
        device: Устройство (CPU/GPU)
        num_classes: Количество классов
    
    Returns:
        Словарь с метриками и детальным отчетом
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    import torch
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    metrics = calculate_metrics(all_targets, all_preds)
    
    class_names = [f'Class_{i}' for i in range(num_classes)]
    report = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        output_dict=True
    )
    
    return {
        'metrics': metrics,
        'classification_report': report
    }

