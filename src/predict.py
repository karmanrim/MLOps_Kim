"""
Скрипт для инференса модели (offline prediction)
Используется в Docker контейнере для пакетной обработки
"""
import argparse
import os
import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.models.cnn_model import create_model_from_config


def load_model(model_path: str, device: str = 'cpu'):
    """
    Загрузить модель с диска
    
    Args:
        model_path: Путь к директории с моделью
        device: Устройство для инференса
        
    Returns:
        Загруженная модель
    """
    print(f"Загрузка модели из {model_path}...")
    
    config_path = os.path.join(model_path, 'config.json')
    with open(config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    model = create_model_from_config(model_config)
    
    weights_file = os.path.join(model_path, 'model.safetensors')
    if not os.path.exists(weights_file):
        weights_file = os.path.join(model_path, 'pytorch_model.bin')
    
    if os.path.exists(weights_file):
        if weights_file.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(weights_file)
        else:
            state_dict = torch.load(weights_file, map_location=device)
        model.load_state_dict(state_dict)
        print("Модель загружена успешно!")
    else:
        raise FileNotFoundError(f"Файл весов модели не найден: {weights_file}")
    
    model = model.to(device)
    model.eval()
    
    return model


def load_images_from_directory(input_path: str):
    """
    Загрузить изображения из директории
    
    Args:
        input_path: Путь к директории с изображениями
        
    Returns:
        images: Список PIL изображений
        filenames: Список имен файлов
    """
    print(f"Загрузка изображений из {input_path}...")
    
    images = []
    filenames = []
    
    supported_formats = ['.png', '.jpg', '.jpeg', '.bmp']
    
    for filename in sorted(os.listdir(input_path)):
        if any(filename.lower().endswith(fmt) for fmt in supported_formats):
            img_path = os.path.join(input_path, filename)
            try:
                img = Image.open(img_path).convert('L')  # Grayscale
                images.append(img)
                filenames.append(filename)
            except Exception as e:
                print(f"Ошибка при загрузке {filename}: {e}")
    
    print(f"Загружено {len(images)} изображений")
    return images, filenames


def preprocess_images(images, target_size=(28, 28)):
    """
    Предобработка изображений для модели
    
    Args:
        images: Список PIL изображений
        target_size: Целевой размер изображений
        
    Returns:
        Тензор изображений
    """
    processed = []
    
    for img in images:
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        
        img_array = np.expand_dims(img_array, axis=0)
        
        processed.append(img_array)
    
    batch = np.stack(processed, axis=0)
    tensor = torch.from_numpy(batch)
    
    return tensor


def predict_batch(model, images_tensor, device, batch_size=32):
    """
    Выполнить предсказания на батче изображений
    
    Args:
        model: Обученная модель
        images_tensor: Тензор изображений
        device: Устройство для вычислений
        batch_size: Размер батча
        
    Returns:
        predictions: Список предсказанных классов
        probabilities: Список вероятностей
    """
    print("Выполнение предсказаний...")
    
    all_preds = []
    all_probs = []
    
    num_samples = images_tensor.size(0)
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = images_tensor[i:i+batch_size].to(device)
            outputs = model(batch)
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    print(f"Предсказания выполнены для {num_samples} изображений")
    return all_preds, all_probs


def save_predictions(predictions, probabilities, filenames, output_path, class_names=None):
    """
    Сохранить предсказания в CSV файл
    
    Args:
        predictions: Список предсказанных классов
        probabilities: Список вероятностей для каждого класса
        filenames: Список имен файлов
        output_path: Путь для сохранения результатов
        class_names: Названия классов (опционально)
    """
    print(f"Сохранение результатов в {output_path}...")
    
    if class_names is None:
        class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
    
    data = {
        'filename': filenames,
        'predicted_class': predictions,
        'predicted_label': [class_names[pred] for pred in predictions],
        'confidence': [probs[pred] for pred, probs in zip(predictions, probabilities)]
    }
    
    probs_array = np.array(probabilities)
    for i, class_name in enumerate(class_names):
        data[f'prob_{class_name}'] = probs_array[:, i]
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"Результаты сохранены в {output_path}")
    print(f"\nПример предсказаний:")
    print(df[['filename', 'predicted_label', 'confidence']].head())


def main():
    """
    Главная функция для запуска инференса
    """
    parser = argparse.ArgumentParser(description='Offline inference для классификации изображений')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Путь к директории с изображениями для предсказания')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Путь для сохранения результатов (CSV файл)')
    parser.add_argument('--model_path', type=str, default='models/best_model',
                        help='Путь к директории с моделью (по умолчанию: models/best_model)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Размер батча для инференса (по умолчанию: 32)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Устройство для вычислений (по умолчанию: cpu)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("OFFLINE INFERENCE - Fashion-MNIST Classification")
    print("="*60)
    print(f"Input path: {args.input_path}")
    print(f"Output path: {args.output_path}")
    print(f"Model path: {args.model_path}")
    print(f"Device: {args.device}")
    print("="*60)
    
    if not os.path.exists(args.input_path):
        print(f"Ошибка: Директория {args.input_path} не существует!")
        sys.exit(1)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device)
    
    images, filenames = load_images_from_directory(args.input_path)
    
    if len(images) == 0:
        print("Ошибка: Не найдено изображений для обработки!")
        sys.exit(1)
    
    images_tensor = preprocess_images(images)
    
    predictions, probabilities = predict_batch(model, images_tensor, device, args.batch_size)
    
    save_predictions(predictions, probabilities, filenames, args.output_path)
    
    print("="*60)
    print("Инференс завершён успешно!")
    print("="*60)


if __name__ == '__main__':
    main()

