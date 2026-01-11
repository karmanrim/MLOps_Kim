"""
Скрипт для подготовки данных (prepare stage)
"""
import argparse
import yaml
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import preprocess_data
from src.data.validation import validate_dataset


def main():
    """Основная функция для подготовки данных"""
    parser = argparse.ArgumentParser(description='Подготовка данных для обучения')
    parser.add_argument('--config', type=str, required=True, help='Путь к конфигурационному файлу')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Загрузка и подготовка данных {config['data']['dataset_name']}...")
    
    train_loader, val_loader, test_loader = preprocess_data(config)
    
    print(f"Данные подготовлены успешно:")
    print(f"  - Обучающая выборка: {len(train_loader.dataset)} образцов")
    print(f"  - Валидационная выборка: {len(val_loader.dataset)} образцов")
    print(f"  - Тестовая выборка: {len(test_loader.dataset)} образцов")
    
    print("\nВалидация данных...")
    validate_dataset(train_loader.dataset)
    print("Валидация данных завершена успешно!")
    
    data_info = {
        'train_size': len(train_loader.dataset),
        'val_size': len(val_loader.dataset),
        'test_size': len(test_loader.dataset),
        'batch_size': config['data']['batch_size'],
        'num_classes': config['model']['num_classes']
    }
    
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/data_info.yaml', 'w') as f:
        yaml.dump(data_info, f)
    
    print(f"\nИнформация о данных сохранена в metrics/data_info.yaml")


if __name__ == '__main__':
    main()

