"""
Тесты для pipeline обучения
"""
import pytest
import torch
import tempfile
import os
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.training.utils import (
    load_config,
    set_random_seed,
    get_device,
    create_optimizer,
    create_scheduler
)
from src.models.cnn_model import create_model_from_config
from src.training.train import train_epoch, validate_epoch


class TestTrainingUtils:
    """Тесты для утилит обучения"""
    
    def test_load_config(self):
        """Тест загрузки конфигурации"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'data': {'batch_size': 32},
                'model': {'num_classes': 10},
                'training': {'num_epochs': 5}
            }
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            assert config['data']['batch_size'] == 32
            assert config['model']['num_classes'] == 10
            assert config['training']['num_epochs'] == 5
        finally:
            os.unlink(config_path)
    
    def test_load_config_not_found(self):
        """Тест загрузки несуществующего конфига"""
        with pytest.raises(FileNotFoundError):
            load_config('nonexistent_config.yaml')
    
    def test_set_random_seed(self):
        """Тест установки random seed"""
        set_random_seed(42)
        tensor1 = torch.randn(10)
        
        set_random_seed(42)
        tensor2 = torch.randn(10)
        
        assert torch.allclose(tensor1, tensor2)
    
    def test_get_device(self):
        """Тест получения устройства"""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda']
    
    def test_create_optimizer_adam(self):
        """Тест создания оптимизатора Adam"""
        model = torch.nn.Linear(10, 5)
        config = {'optimizer': 'Adam', 'learning_rate': 0.001, 'weight_decay': 0.0001}
        
        optimizer = create_optimizer(model, config)
        assert isinstance(optimizer, torch.optim.Adam)
    
    def test_create_optimizer_sgd(self):
        """Тест создания оптимизатора SGD"""
        model = torch.nn.Linear(10, 5)
        config = {'optimizer': 'SGD', 'learning_rate': 0.01, 'momentum': 0.9}
        
        optimizer = create_optimizer(model, config)
        assert isinstance(optimizer, torch.optim.SGD)
    
    def test_create_optimizer_unknown(self):
        """Тест создания неизвестного оптимизатора"""
        model = torch.nn.Linear(10, 5)
        config = {'optimizer': 'Unknown'}
        
        with pytest.raises(ValueError):
            create_optimizer(model, config)
    
    def test_create_scheduler_steplr(self):
        """Тест создания scheduler StepLR"""
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        config = {'scheduler': 'StepLR', 'step_size': 5, 'gamma': 0.1}
        
        scheduler = create_scheduler(optimizer, config)
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    
    def test_create_scheduler_none(self):
        """Тест создания scheduler когда он не указан"""
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        config = {}
        
        scheduler = create_scheduler(optimizer, config)
        assert scheduler is None


class TestTrainingPipeline:
    """Тесты для pipeline обучения"""
    
    @pytest.fixture
    def sample_model(self):
        """Фикстура для создания тестовой модели"""
        model_config = {'num_classes': 10, 'input_size': 28, 'hidden_dims': [64, 32], 'dropout': 0.5}
        return create_model_from_config(model_config)
    
    @pytest.fixture
    def sample_dataloader(self):
        """Фикстура для создания тестового DataLoader"""
        data = torch.randn(100, 1, 28, 28)
        target = torch.randint(0, 10, (100,))
        dataset = TensorDataset(data, target)
        return DataLoader(dataset, batch_size=32)
    
    def test_train_epoch(self, sample_model, sample_dataloader):
        """Тест обучения на одной эпохе"""
        device = torch.device('cpu')
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(sample_model.parameters(), lr=0.001)
        
        import logging
        logger = logging.getLogger(__name__)
        
        loss, acc = train_epoch(sample_model, sample_dataloader, criterion, optimizer, device, logger)
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= 100
    
    def test_validate_epoch(self, sample_model, sample_dataloader):
        """Тест валидации на одной эпохе"""
        device = torch.device('cpu')
        criterion = torch.nn.CrossEntropyLoss()
        
        import logging
        logger = logging.getLogger(__name__)
        
        loss, acc, preds, targets = validate_epoch(sample_model, sample_dataloader, criterion, device, logger)
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= 100
        assert len(preds) == len(targets)
        assert all(0 <= p < 10 for p in preds)
        assert all(0 <= t < 10 for t in targets)
    
    def test_training_pipeline_reproducibility(self, sample_model, sample_dataloader):
        """Тест воспроизводимости обучения"""
        device = torch.device('cpu')
        criterion = torch.nn.CrossEntropyLoss()
        
        import logging
        logger = logging.getLogger(__name__)
        
        set_random_seed(42)
        model1 = create_model_from_config({'num_classes': 10, 'input_size': 28, 'hidden_dims': [64, 32], 'dropout': 0.5})
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
        loss1, acc1 = train_epoch(model1, sample_dataloader, criterion, optimizer1, device, logger)
        
        set_random_seed(42)
        model2 = create_model_from_config({'num_classes': 10, 'input_size': 28, 'hidden_dims': [64, 32], 'dropout': 0.5})
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        loss2, acc2 = train_epoch(model2, sample_dataloader, criterion, optimizer2, device, logger)
        
        assert abs(loss1 - loss2) < 1e-5
        assert abs(acc1 - acc2) < 1e-5

