"""
Скрипт для оценки модели (evaluate stage)
"""
import argparse
import yaml
import os
import sys
import json
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import preprocess_data
from src.models.cnn_model import create_model_from_config
from src.training.metrics import calculate_metrics
from src.training.utils import load_config, get_device


def evaluate_model(model, test_loader, device, config):
    """
    Оценка модели на тестовой выборке
    
    Args:
        model: Модель для оценки
        test_loader: DataLoader для тестовых данных
        device: Устройство для вычислений
        config: Конфигурация
        
    Returns:
        dict: Метрики модели
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    print("Оценка модели на тестовой выборке...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['test_loss'] = avg_loss
    
    return metrics


def main():
    """Основная функция для оценки модели"""
    parser = argparse.ArgumentParser(description='Оценка обученной модели')
    parser.add_argument('--config', type=str, required=True, help='Путь к конфигурационному файлу')
    parser.add_argument('--model-path', type=str, default='models/best_model', 
                        help='Путь к обученной модели')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = get_device()
    
    print(f"Загрузка модели из {args.model_path}...")
    
    model = create_model_from_config(config)
    
    model_file = os.path.join(args.model_path, 'model.safetensors')
    if not os.path.exists(model_file):
        model_file = os.path.join(args.model_path, 'pytorch_model.bin')
    
    if os.path.exists(model_file):
        if model_file.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(model_file)
        else:
            state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Модель загружена успешно!")
    else:
        raise FileNotFoundError(f"Файл модели не найден: {model_file}")
    
    model = model.to(device)
    
    print("Загрузка тестовых данных...")
    _, _, test_loader = preprocess_data(config)
    
    metrics = evaluate_model(model, test_loader, device, config)
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ")
    print("="*50)
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print("="*50)
    
    os.makedirs('metrics', exist_ok=True)
    
    with open('metrics/test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open('metrics/test_metrics.yaml', 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    
    print(f"\nМетрики сохранены в metrics/test_metrics.json и metrics/test_metrics.yaml")
    
    target_accuracy = 0.90
    target_f1 = 0.88
    
    if metrics['accuracy'] >= target_accuracy and metrics['f1'] >= target_f1:
        print(f"\n✓ Модель достигла целевых метрик!")
        print(f"  Accuracy >= {target_accuracy}: {metrics['accuracy']:.4f}")
        print(f"  F1-Score >= {target_f1}: {metrics['f1']:.4f}")
    else:
        print(f"\n✗ Модель не достигла целевых метрик:")
        if metrics['accuracy'] < target_accuracy:
            print(f"  Accuracy < {target_accuracy}: {metrics['accuracy']:.4f}")
        if metrics['f1'] < target_f1:
            print(f"  F1-Score < {target_f1}: {metrics['f1']:.4f}")


if __name__ == '__main__':
    main()

