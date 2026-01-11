"""
Скрипт для экспорта модели в формат для TorchServe
Сохраняет модель в state_dict формате
"""
import os
import sys
from pathlib import Path
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.models.cnn_model import create_model_from_config


def export_model_for_torchserve(model_path: str, output_path: str):
    """
    Экспорт модели для TorchServe
    
    Args:
        model_path: Путь к сохранённой модели (Hugging Face формат)
        output_path: Путь для сохранения модели в PyTorch формате
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
            state_dict = torch.load(weights_file, map_location='cpu')
        model.load_state_dict(state_dict)
        print("Веса модели загружены успешно!")
    else:
        raise FileNotFoundError(f"Файл весов не найден: {weights_file}")
    
    model.eval()
    
    print(f"Сохранение модели в {output_path}...")
    torch.save(model.state_dict(), output_path)
    print("Модель успешно экспортирована для TorchServe!")
    
    # Также сохраняем конфигурацию рядом
    config_output = output_path.replace('.pt', '_config.json')
    with open(config_output, 'w') as f:
        yaml.dump(model_config, f)
    print(f"Конфигурация сохранена в {config_output}")
    
    return output_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Экспорт модели для TorchServe')
    parser.add_argument('--model-path', type=str, default='models/best_model',
                        help='Путь к модели (по умолчанию: models/best_model)')
    parser.add_argument('--output-path', type=str, default='torchserve/model.pt',
                        help='Путь для сохранения (по умолчанию: torchserve/model.pt)')
    
    args = parser.parse_args()
    
    export_model_for_torchserve(args.model_path, args.output_path)

