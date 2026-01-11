#!/bin/bash
# Пример запуска Docker контейнера для инференса

# Сборка образа
echo "Сборка Docker образа..."
docker build -t ml-app:v1 .

# Проверка что образ собран
if [ $? -eq 0 ]; then
    echo "✓ Образ ml-app:v1 успешно собран"
else
    echo "✗ Ошибка при сборке образа"
    exit 1
fi

# Создание тестовой директории если нужно
mkdir -p test_images
mkdir -p predictions

# Запуск контейнера для инференса
echo ""
echo "Запуск контейнера для инференса..."
docker run --rm \
    -v $(pwd)/test_images:/app/input:ro \
    -v $(pwd)/predictions:/app/output \
    ml-app:v1 \
    --input_path /app/input \
    --output_path /app/output/predictions.csv \
    --model_path /app/models/best_model \
    --batch_size 32 \
    --device cpu

# Проверка результатов
if [ -f predictions/predictions.csv ]; then
    echo ""
    echo "✓ Предсказания сохранены в predictions/predictions.csv"
    echo ""
    echo "Первые строки результатов:"
    head -n 10 predictions/predictions.csv
else
    echo "✗ Файл с предсказаниями не найден"
fi

