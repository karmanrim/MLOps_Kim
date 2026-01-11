#!/bin/bash
# Скрипт для запуска TorchServe сервиса

set -e

echo "========================================="
echo "TorchServe Deployment Script"
echo "========================================="

# Шаг 1: Экспорт модели
echo ""
echo "[1/4] Exporting model to TorchServe format..."
if [ ! -f "torchserve/model.pt" ]; then
    conda activate mlops_roma && python export_model_for_torchserve.py \
        --model-path models/best_model \
        --output-path torchserve/model.pt
    echo "✓ Model exported successfully"
else
    echo "✓ Model already exported (torchserve/model.pt exists)"
fi

# Шаг 2: Создание MAR архива
echo ""
echo "[2/4] Creating MAR archive..."
if [ ! -f "torchserve/model-store/fashion_mnist.mar" ]; then
    ./create_mar_archive.sh
else
    echo "✓ MAR archive already exists"
fi

# Шаг 3: Сборка Docker образа
echo ""
echo "[3/4] Building Docker image..."
cd torchserve
docker build -t mymodel-serve:v1 .
cd ..

if [ $? -eq 0 ]; then
    echo "✓ Docker image built successfully"
else
    echo "✗ Failed to build Docker image"
    exit 1
fi

# Шаг 4: Запуск контейнера
echo ""
echo "[4/4] Starting TorchServe container..."

# Остановка старого контейнера если есть
docker stop torchserve-container 2>/dev/null || true
docker rm torchserve-container 2>/dev/null || true

# Запуск нового контейнера
docker run -d \
    --name torchserve-container \
    -p 8080:8080 \
    -p 8081:8081 \
    -p 8082:8082 \
    mymodel-serve:v1

if [ $? -eq 0 ]; then
    echo "✓ TorchServe container started successfully"
    echo ""
    echo "========================================="
    echo "Service is running!"
    echo "========================================="
    echo "Inference API: http://localhost:8080"
    echo "Management API: http://localhost:8081"
    echo "Metrics API: http://localhost:8082"
    echo ""
    echo "Wait a few seconds for the service to initialize..."
    echo ""
    
    # Ожидание инициализации
    sleep 5
    
    # Проверка здоровья сервиса
    echo "Checking service health..."
    curl -s http://localhost:8080/ping || echo "Service not ready yet, please wait..."
    
    echo ""
    echo "To test the service:"
    echo "  curl -X POST http://localhost:8080/predictions/fashion_mnist -T torchserve/sample_input.json"
    echo ""
    echo "To stop the service:"
    echo "  docker stop torchserve-container"
else
    echo "✗ Failed to start container"
    exit 1
fi

