"""
Custom Handler для TorchServe
Обрабатывает предсказания для Fashion-MNIST классификации
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import json
import base64
import logging

logger = logging.getLogger(__name__)


class FashionMNISTHandler:
    """
    Custom handler для Fashion-MNIST модели
    """
    
    def __init__(self):
        self.model = None
        self.device = None
        self.initialized = False
        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            # Fashion-MNIST нормализация
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def initialize(self, context):
        """
        Инициализация модели
        
        Args:
            context: TorchServe context с информацией о модели
        """
        logger.info("Initializing Fashion-MNIST model...")
        
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id")) 
            if torch.cuda.is_available() 
            else "cpu"
        )
        
        manifest = context.manifest
        
        serialized_file = manifest['model']['serializedFile']
        model_pt_path = f"{model_dir}/{serialized_file}"
        
        self.model = self._create_model()
        
        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.initialized = True
        logger.info("Model initialized successfully!")
    
    def _create_model(self):
        """Создание архитектуры модели"""
        class CNNModel(nn.Module):
            def __init__(self, num_classes=10):
                super(CNNModel, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout1 = nn.Dropout(0.25)
                self.fc1 = nn.Linear(64 * 7 * 7, 128)
                self.dropout2 = nn.Dropout(0.5)
                self.fc2 = nn.Linear(128, num_classes)
            
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = self.dropout1(x)
                x = x.view(-1, 64 * 7 * 7)
                x = F.relu(self.fc1(x))
                x = self.dropout2(x)
                x = self.fc2(x)
                return x
        
        return CNNModel(num_classes=10)
    
    def preprocess(self, data):
        """
        Предобработка входных данных
        
        Args:
            data: Список запросов, каждый может содержать:
                  - base64 encoded image
                  - binary image data
                  
        Returns:
            Тензор изображений
        """
        images = []
        
        for row in data:
            image = row.get("data") or row.get("body")
            
            if isinstance(image, str):
                # Base64 encoded image
                image = base64.b64decode(image)
            
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
            elif isinstance(image, dict):
                # JSON с base64
                if "image" in image:
                    image_data = base64.b64decode(image["image"])
                    image = Image.open(io.BytesIO(image_data))
            
            # Применение трансформаций
            image_tensor = self.transform(image)
            images.append(image_tensor)
        
        # Stack в batch
        return torch.stack(images).to(self.device)
    
    def inference(self, model_input):
        """
        Выполнение инференса
        
        Args:
            model_input: Предобработанный тензор
            
        Returns:
            Предсказания модели
        """
        with torch.no_grad():
            outputs = self.model(model_input)
            return outputs
    
    def postprocess(self, inference_output):
        """
        Постобработка предсказаний
        
        Args:
            inference_output: Выход модели (логиты)
            
        Returns:
            Список словарей с результатами
        """
        # Вычисление вероятностей
        probabilities = F.softmax(inference_output, dim=1)
        
        # Получение предсказанных классов
        predicted_classes = torch.argmax(probabilities, dim=1)
        
        # Формирование ответа
        results = []
        
        for i in range(len(predicted_classes)):
            pred_class = predicted_classes[i].item()
            pred_label = self.class_names[pred_class]
            confidence = probabilities[i][pred_class].item()
            
            # Вероятности для всех классов
            class_probabilities = {
                self.class_names[j]: probabilities[i][j].item()
                for j in range(len(self.class_names))
            }
            
            result = {
                "predicted_class": pred_class,
                "predicted_label": pred_label,
                "confidence": round(confidence, 4),
                "probabilities": class_probabilities
            }
            
            results.append(result)
        
        return results
    
    def handle(self, data, context):
        """
        Главная функция обработки запроса
        
        Args:
            data: Входные данные
            context: Контекст TorchServe
            
        Returns:
            Результаты предсказания
        """
        if not self.initialized:
            self.initialize(context)
        
        # Предобработка
        model_input = self.preprocess(data)
        
        # Инференс
        model_output = self.inference(model_input)
        
        # Постобработка
        return self.postprocess(model_output)


# Глобальный экземпляр handler
_service = FashionMNISTHandler()


def handle(data, context):
    """Entry point для TorchServe"""
    return _service.handle(data, context)

