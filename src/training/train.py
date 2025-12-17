"""
Скрипт для обучения модели
"""
import argparse
import yaml
import os
import sys
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import preprocess_data
from src.models.cnn_model import create_model_from_config
from src.training.metrics import calculate_metrics, validate_model


def setup_logging(log_dir: str, verbose: bool = False):
    """
    Настройка логирования
    
    Args:
        log_dir: Директория для логов
        verbose: Подробный режим
    """
    os.makedirs(log_dir, exist_ok=True)
    
    log_level = logging.DEBUG if verbose else logging.INFO
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    file_handler = logging.FileHandler(
        os.path.join(log_dir, 'training.log'),
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger


def train_epoch(model, train_loader, criterion, optimizer, device, logger):
    """
    Обучение модели на одной эпохе
    
    Args:
        model: Модель для обучения
        train_loader: DataLoader для обучающих данных
        criterion: Функция потерь
        optimizer: Оптимизатор
        device: Устройство (CPU/GPU)
        logger: Логгер
    
    Returns:
        Средняя потеря и точность на эпохе
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device, logger):
    """
    Валидация модели на одной эпохе
    
    Args:
        model: Модель для валидации
        val_loader: DataLoader для валидационных данных
        criterion: Функция потерь
        device: Устройство (CPU/GPU)
        logger: Логгер
    
    Returns:
        Средняя потеря и точность на эпохе
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_targets


def train(config_path: str, verbose: bool = False):
    """
    Основная функция обучения
    
    Args:
        config_path: Путь к конфигурационному файлу
        verbose: Подробный режим логирования
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    log_dir = config.get('logging', {}).get('log_dir', 'logs')
    logger = setup_logging(log_dir, verbose)
    logger.info("=" * 50)
    logger.info("Начало обучения модели")
    logger.info("=" * 50)
    logger.info(f"Конфигурация загружена из: {config_path}")
    
    random_seed = config.get('training', {}).get('random_seed', 42)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    logger.info(f"Random seed установлен: {random_seed}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используемое устройство: {device}")
    
    logger.info("Загрузка и предобработка данных...")
    data_config = config.get('data', {})
    data_config['random_seed'] = random_seed
    train_loader, val_loader, test_loader = preprocess_data(data_config)
    logger.info("Данные загружены успешно")
    
    logger.info("Создание модели...")
    model_config = config.get('model', {})
    model = create_model_from_config(model_config)
    model = model.to(device)
    logger.info(f"Модель создана и перемещена на {device}")
    
    criterion = nn.CrossEntropyLoss()
    
    training_config = config.get('training', {})
    learning_rate = training_config.get('learning_rate', 0.001)
    weight_decay = training_config.get('weight_decay', 0.0001)
    optimizer_name = training_config.get('optimizer', 'Adam')
    
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        momentum = training_config.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Неизвестный оптимизатор: {optimizer_name}")
    
    logger.info(f"Оптимизатор: {optimizer_name}, LR: {learning_rate}, Weight decay: {weight_decay}")
    
    scheduler = None
    scheduler_type = training_config.get('scheduler', None)
    if scheduler_type == 'StepLR':
        step_size = training_config.get('step_size', 10)
        gamma = training_config.get('gamma', 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        logger.info(f"Scheduler: StepLR (step_size={step_size}, gamma={gamma})")
    
    num_epochs = training_config.get('num_epochs', 10)
    best_val_acc = 0.0
    save_dir = config.get('model', {}).get('save_dir', 'models')
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Начало обучения на {num_epochs} эпох")
    logger.info("-" * 50)
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"Эпоха {epoch}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, logger)
        logger.info(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device, logger)
        logger.info(f"Val - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        metrics = calculate_metrics(val_targets, val_preds)
        logger.info(f"Val Metrics - Precision: {metrics['precision']:.4f}, "
                   f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        
        if scheduler is not None:
            scheduler.step()
            logger.info(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(save_dir, 'best_model')
            model.save_pretrained(model_path)
            logger.info(f"Лучшая модель сохранена в {model_path} (Val Acc: {val_acc:.2f}%)")
        
        logger.info("-" * 50)
    
    logger.info("Финальная валидация на тестовой выборке...")
    test_loss, test_acc, test_preds, test_targets = validate_epoch(model, test_loader, criterion, device, logger)
    test_metrics = calculate_metrics(test_targets, test_preds)
    
    logger.info("=" * 50)
    logger.info("РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ:")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test F1-score: {test_metrics['f1']:.4f}")
    logger.info("=" * 50)
    
    final_model_path = os.path.join(save_dir, 'final_model')
    model.save_pretrained(final_model_path)
    logger.info(f"Финальная модель сохранена в {final_model_path}")
    
    logger.info("Обучение завершено!")


def main():
    """Точка входа"""
    parser = argparse.ArgumentParser(description='Обучение модели классификации изображений')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Путь к конфигурационному файлу'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Подробный режим логирования'
    )
    
    args = parser.parse_args()
    
    train(args.config, verbose=args.verbose)


if __name__ == '__main__':
    main()

