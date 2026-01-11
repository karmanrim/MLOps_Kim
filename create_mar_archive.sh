#!/bin/bash
# Скрипт для создания MAR архива для TorchServe

set -e

echo "========================================="
echo "Creating TorchServe Model Archive (MAR)"
echo "========================================="

# Проверка что model.pt существует
if [ ! -f "torchserve/model.pt" ]; then
    echo "Error: torchserve/model.pt not found!"
    echo "Please run: python export_model_for_torchserve.py first"
    exit 1
fi

# Проверка что handler.py существует  
if [ ! -f "torchserve/handler.py" ]; then
    echo "Error: torchserve/handler.py not found!"
    exit 1
fi

# Создание директории model-store если не существует
mkdir -p torchserve/model-store

echo ""
echo "Creating MAR archive..."

# Создание MAR архива
torch-model-archiver \
    --model-name fashion_mnist \
    --version 1.0 \
    --serialized-file torchserve/model.pt \
    --handler torchserve/handler.py \
    --export-path torchserve/model-store \
    --force

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ MAR archive created successfully!"
    echo "  Location: torchserve/model-store/fashion_mnist.mar"
    echo ""
    ls -lh torchserve/model-store/fashion_mnist.mar
else
    echo ""
    echo "✗ Failed to create MAR archive"
    exit 1
fi

echo ""
echo "========================================="
echo "Archive ready for TorchServe deployment"
echo "========================================="

