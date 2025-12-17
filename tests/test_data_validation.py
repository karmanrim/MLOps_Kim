"""
Тесты для валидации данных
"""
import pytest
import torch
from src.data.validation import (
    validate_data_format,
    validate_data_types,
    validate_data_features,
    validate_target_labels,
    validate_dataset
)
from torch.utils.data import DataLoader, TensorDataset


class TestDataValidation:
    """Тесты для валидации данных"""
    
    def test_validate_data_format_correct(self):
        """Тест корректного формата данных"""
        data = torch.randn(32, 1, 28, 28)
        target = torch.randint(0, 10, (32,))
        
        is_valid, error = validate_data_format(data, target)
        assert is_valid, error
    
    def test_validate_data_format_wrong_type(self):
        """Тест неправильного типа данных"""
        data = [[1, 2, 3]]
        target = torch.randint(0, 10, (32,))
        
        is_valid, error = validate_data_format(data, target)
        assert not is_valid
        assert "torch.Tensor" in error
    
    def test_validate_data_format_wrong_shape(self):
        """Тест неправильной формы данных"""
        data = torch.randn(32, 28, 28)
        target = torch.randint(0, 10, (32,))
        
        is_valid, error = validate_data_format(data, target)
        assert not is_valid
        assert "форма" in error or "shape" in error.lower()
    
    def test_validate_data_format_batch_mismatch(self):
        """Тест несовпадения размеров батча"""
        data = torch.randn(32, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        
        is_valid, error = validate_data_format(data, target)
        assert not is_valid
        assert "размер батча" in error or "batch" in error.lower()
    
    def test_validate_data_types_correct(self):
        """Тест корректных типов данных"""
        data = torch.randn(32, 1, 28, 28).float()
        data = (data - data.min()) / (data.max() - data.min()) * 2 - 1
        target = torch.randint(0, 10, (32,)).long()
        
        is_valid, error = validate_data_types(data, target)
        assert is_valid, error
    
    def test_validate_data_types_wrong_dtype(self):
        """Тест неправильного типа данных"""
        data = torch.randn(32, 1, 28, 28).double()
        target = torch.randint(0, 10, (32,)).long()
        
        is_valid, error = validate_data_types(data, target)
        assert not is_valid
        assert "float32" in error
    
    def test_validate_data_types_wrong_range(self):
        """Тест данных вне допустимого диапазона"""
        data = torch.randn(32, 1, 28, 28) * 10
        target = torch.randint(0, 10, (32,)).long()
        
        is_valid, error = validate_data_types(data, target)
        assert not is_valid
        assert "диапазон" in error or "range" in error.lower()
    
    def test_validate_data_features_correct(self):
        """Тест корректных признаков"""
        data = torch.randn(32, 1, 28, 28)
        
        is_valid, error = validate_data_features(data, expected_channels=1, expected_size=28)
        assert is_valid, error
    
    def test_validate_data_features_wrong_channels(self):
        """Тест неправильного количества каналов"""
        data = torch.randn(32, 3, 28, 28)
        
        is_valid, error = validate_data_features(data, expected_channels=1, expected_size=28)
        assert not is_valid
        assert "канал" in error or "channel" in error.lower()
    
    def test_validate_data_features_wrong_size(self):
        """Тест неправильного размера изображения"""
        data = torch.randn(32, 1, 32, 32)
        
        is_valid, error = validate_data_features(data, expected_channels=1, expected_size=28)
        assert not is_valid
        assert "размер" in error or "size" in error.lower()
    
    def test_validate_target_labels_correct(self):
        """Тест корректных меток"""
        target = torch.randint(0, 10, (32,))
        
        is_valid, error = validate_target_labels(target, num_classes=10)
        assert is_valid, error
    
    def test_validate_target_labels_out_of_range(self):
        """Тест меток вне допустимого диапазона"""
        target = torch.randint(10, 20, (32,))
        
        is_valid, error = validate_target_labels(target, num_classes=10)
        assert not is_valid
        assert "меньше" in error or "less" in error.lower()
    
    def test_validate_dataset_correct(self):
        """Тест валидации полного датасета"""
        data = torch.randn(100, 1, 28, 28).float()
        data = (data - data.min()) / (data.max() - data.min()) * 2 - 1
        target = torch.randint(0, 10, (100,)).long()
        
        dataset = TensorDataset(data, target)
        dataloader = DataLoader(dataset, batch_size=32)
        
        is_valid, errors = validate_dataset(dataloader, num_classes=10, expected_size=28)
        assert is_valid, errors

