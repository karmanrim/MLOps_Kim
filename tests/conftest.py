"""
Конфигурация для pytest
"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def project_root_path():
    """Фикстура для получения пути к корню проекта"""
    return Path(__file__).parent.parent

