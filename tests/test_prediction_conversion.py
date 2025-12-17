"""
Тесты для конвертации предсказаний
"""
import pytest
import torch
import numpy as np
from src.inference.prediction import (
    convert_logits_to_probabilities,
    convert_probabilities_to_class,
    convert_logits_to_class,
    convert_to_numpy,
    process_model_output,
    validate_prediction_output
)


class TestPredictionConversion:
    """Тесты для конвертации предсказаний"""
    
    def test_convert_logits_to_probabilities(self):
        """Тест конвертации логитов в вероятности"""
        logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        probabilities = convert_logits_to_probabilities(logits)
        
        assert isinstance(probabilities, torch.Tensor)
        assert probabilities.shape == logits.shape
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-6)
        assert (probabilities >= 0).all() and (probabilities <= 1).all()
    
    def test_convert_logits_to_probabilities_wrong_type(self):
        """Тест с неправильным типом входных данных"""
        logits = [[1.0, 2.0, 3.0]]
        
        with pytest.raises(TypeError):
            convert_logits_to_probabilities(logits)
    
    def test_convert_probabilities_to_class(self):
        """Тест конвертации вероятностей в класс"""
        probabilities = torch.tensor([[0.1, 0.8, 0.1], [0.5, 0.3, 0.2]])
        classes = convert_probabilities_to_class(probabilities)
        
        assert isinstance(classes, torch.Tensor)
        assert classes.shape == (2,)
        assert classes[0] == 1
        assert classes[1] == 0
    
    def test_convert_probabilities_to_class_wrong_type(self):
        """Тест с неправильным типом входных данных"""
        probabilities = [[0.1, 0.8, 0.1]]
        
        with pytest.raises(TypeError):
            convert_probabilities_to_class(probabilities)
    
    def test_convert_logits_to_class(self):
        """Тест прямой конвертации логитов в класс"""
        logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        classes = convert_logits_to_class(logits)
        
        assert isinstance(classes, torch.Tensor)
        assert classes.shape == (2,)
        assert classes[0] == 2
        assert classes[1] == 0
    
    def test_convert_to_numpy_from_tensor(self):
        """Тест конвертации из torch.Tensor в numpy"""
        tensor = torch.tensor([1, 2, 3])
        numpy_array = convert_to_numpy(tensor)
        
        assert isinstance(numpy_array, np.ndarray)
        assert np.array_equal(numpy_array, np.array([1, 2, 3]))
    
    def test_convert_to_numpy_from_array(self):
        """Тест конвертации из numpy array в numpy (без изменений)"""
        array = np.array([1, 2, 3])
        result = convert_to_numpy(array)
        
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, array)
    
    def test_convert_to_numpy_wrong_type(self):
        """Тест с неправильным типом входных данных"""
        wrong_input = [1, 2, 3]
        
        with pytest.raises(TypeError):
            convert_to_numpy(wrong_input)
    
    def test_process_model_output(self):
        """Тест полной обработки выходов модели"""
        logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        result = process_model_output(logits, return_probabilities=True)
        
        assert 'predicted_class' in result
        assert 'probabilities' in result
        assert isinstance(result['predicted_class'], np.ndarray)
        assert isinstance(result['probabilities'], np.ndarray)
        assert result['predicted_class'].shape == (2,)
        assert result['probabilities'].shape == (2, 3)
    
    def test_process_model_output_without_probabilities(self):
        """Тест обработки выходов без вероятностей"""
        logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        result = process_model_output(logits, return_probabilities=False)
        
        assert 'predicted_class' in result
        assert 'probabilities' not in result
    
    def test_validate_prediction_output_correct(self):
        """Тест валидации корректных выходов"""
        logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        
        is_valid, error = validate_prediction_output(logits)
        assert is_valid, error
    
    def test_validate_prediction_output_wrong_type(self):
        """Тест валидации с неправильным типом"""
        logits = [[1.0, 2.0, 3.0]]
        
        is_valid, error = validate_prediction_output(logits)
        assert not is_valid
        assert "torch.Tensor" in error
    
    def test_validate_prediction_output_wrong_shape(self):
        """Тест валидации с неправильной формой"""
        logits = torch.tensor([1.0, 2.0, 3.0])
        
        is_valid, error = validate_prediction_output(logits)
        assert not is_valid
        assert "форма" in error or "shape" in error.lower()
    
    def test_validate_prediction_output_nan(self):
        """Тест валидации с NaN значениями"""
        logits = torch.tensor([[1.0, 2.0, float('nan')]])
        
        is_valid, error = validate_prediction_output(logits)
        assert not is_valid
        assert "нефинитные" in error or "finite" in error.lower()

