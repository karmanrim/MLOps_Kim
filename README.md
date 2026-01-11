# ML-проект: Автоматическая категоризация товаров по изображениям

## Цель проекта

**Бизнес-цель:** Разработка системы автоматической категоризации товаров в интернет-магазине на основе изображений для ускорения процесса добавления новых товаров в каталог и улучшения пользовательского опыта поиска. Система позволит автоматически определять категорию товара по его изображению, что сократит время на ручную категоризацию и уменьшит количество ошибок.

**Техническая цель:** Создать модель глубокого обучения, способную классифицировать изображения товаров по категориям с высокой точностью, и развернуть её в продакшене с соблюдением требований по производительности и надежности.

## Набор данных

Для обучения модели будет использован датасет **Fashion-MNIST** или расширенный датасет с изображениями товаров различных категорий.

### Характеристики датасета:
- **Тип данных:** Изображения товаров (одежда, обувь, аксессуары и т.д.)
- **Размер:** ~70,000 изображений для обучения, ~10,000 для валидации
- **Формат:** Градации серого изображения размером 28x28 пикселей
- **Классы:** 10 категорий товаров:
  - T-shirt/top (Футболка)
  - Trouser (Брюки)
  - Pullover (Свитер)
  - Dress (Платье)
  - Coat (Пальто)
  - Sandal (Сандалии)
  - Shirt (Рубашка)
  - Sneaker (Кроссовки)
  - Bag (Сумка)
  - Ankle boot (Ботинки)

Датасет будет загружаться автоматически при первом запуске через PyTorch или может быть предоставлен в виде архивного файла.

## План экспериментов

### Этап 1: Подготовка данных и базовая модель
- Загрузка и анализ датасета Fashion-MNIST
- Предобработка данных (нормализация, аугментация)
- Создание базовой CNN модели
- Обучение на небольшой выборке для проверки pipeline
- Базовая валидация и проверка работоспособности

**Ожидаемый результат:** Рабочий pipeline обучения с базовой моделью, дающей accuracy > 80%

### Этап 2: Оптимизация модели
- Эксперименты с различными архитектурами:
  - Базовая CNN
  - ResNet (адаптированная для Fashion-MNIST)
  - EfficientNet (если возможно масштабирование)
- Подбор гиперпараметров:
  - Learning rate (grid search: 0.0001, 0.001, 0.01)
  - Batch size (32, 64, 128)
  - Optimizer (Adam, SGD, AdamW)
- Применение техник регуляризации:
  - Dropout (0.3, 0.5, 0.7)
  - Batch normalization
  - Data augmentation (random flip, rotation, brightness)
- Анализ метрик качества:
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrix для анализа ошибок

**Ожидаемый результат:** Модель с accuracy ≥ 90% на тестовой выборке

### Этап 3: Оптимизация для продакшена
- Квантизация модели для уменьшения размера (INT8)
- Оптимизация инференса:
  - Конвертация в ONNX формат
  - Оптимизация с помощью TensorRT (если используется GPU)
- Тестирование производительности:
  - Измерение времени инференса
  - Нагрузочное тестирование API
  - Проверка использования памяти и CPU
- Достижение целевых метрик производительности

**Ожидаемый результат:** Модель, удовлетворяющая всем целевым метрикам производительности

### Этап 4: Развертывание и мониторинг
- Создание API сервиса:
  - FastAPI для REST API
  - Обработка изображений (загрузка, предобработка)
  - Обработка ошибок и валидация входных данных
- Контейнеризация:
  - Docker образ с моделью и API
  - Docker Compose для локального развертывания
- Настройка CI/CD pipeline:
  - Автоматические тесты
  - Автоматическое развертывание
- Внедрение логирования и мониторинга:
  - Логирование запросов и ответов
  - Метрики производительности (Prometheus)
  - Алерты при превышении SLA

**Ожидаемый результат:** Полностью развернутый и мониторируемый сервис в продакшене

## Целевые метрики для продакшена

### Метрики производительности сервиса:
- **Среднее время отклика сервиса ≤ 200 мс** (p95 ≤ 300 мс)
  - Включает время загрузки изображения, предобработки, инференса и формирования ответа
- **Доля неуспешных запросов ≤ 1%** (error rate)
  - Учитываются ошибки сервера (5xx), таймауты, ошибки валидации
- **Использование памяти/CPU — в пределах SLA**
  - Память: ≤ 2GB для модели и сервиса
  - CPU: ≤ 70% при пиковой нагрузке (100 запросов/секунду)
- **Пропускная способность:** ≥ 100 запросов/секунду

### Метрики качества модели:
- **Точность (Accuracy) ≥ 90%** на тестовой выборке
- **Precision и Recall ≥ 85%** для каждой категории
- **F1-score ≥ 0.88** (макро-усредненный)

### Дополнительные метрики:
- **Время инференса модели ≤ 50 мс** (на CPU/GPU)
- **Размер модели ≤ 100 MB** (после оптимизации и квантизации)
- **Время холодного старта сервиса ≤ 5 секунд**

## Структура проекта

```
MLOps_Kim/
├── README.md              # Описание проекта
├── requirements.txt       # Зависимости проекта
├── .gitignore            # Игнорируемые файлы
├── dvc.yaml              # DVC pipeline конфигурация
├── dvc.lock              # DVC pipeline lock файл
├── data.dvc              # DVC файл для данных
├── models.dvc            # DVC файл для моделей
├── data/                 # Данные (версионируются через DVC)
├── models/              # Модели (версионируются через DVC)
├── metrics/             # Метрики экспериментов
├── src/                  # Исходный код
│   ├── data/            # Модули для работы с данными
│   ├── models/          # Архитектуры моделей
│   ├── training/        # Скрипты обучения
│   └── inference/       # Код для инференса
├── notebooks/            # Jupyter notebooks для экспериментов
├── tests/               # Тесты
├── configs/             # Конфигурационные файлы
├── docker/              # Docker файлы для развертывания
└── docs/                # Документация
```

## Технологический стек

- **ML Framework:** PyTorch
- **Версионирование данных и моделей:** DVC
- **Трекинг экспериментов:** MLflow
- **Контейнеризация:** Docker (offline inference)
- **API Framework:** FastAPI (планируется)
- **Мониторинг:** Prometheus, Grafana (опционально)
- **CI/CD:** GitHub Actions
- **Логирование:** Python logging, структурированные логи

## Установка и запуск

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Работа с DVC

#### Первоначальная настройка

После клонирования репозитория необходимо получить данные и модели:

```bash
# Клонирование репозитория
git clone <repository-url>
cd MLOps_Kim

# Получение данных и моделей из удалённого хранилища DVC
dvc pull
```

#### Где физически лежат данные/модели

Данные и модели версионируются через DVC и хранятся в удалённом хранилище. Локально они находятся в директориях:
- `data/` - датасет FashionMNIST
- `models/` - обученные модели (best_model и final_model)

Файлы `.dvc` (data.dvc, models.dvc) содержат метаинформацию и хеши для отслеживания версий.

#### Команды для работы с версиями

```bash
# Получить данные и модели из удалённого хранилища
dvc pull

# Отправить изменения в удалённое хранилище
dvc push

# Переключиться на другую версию данных/моделей
git checkout <commit-hash>
dvc checkout

# Посмотреть статус DVC
dvc status

# Посмотреть DAG пайплайна
dvc dag
```

#### План экспериментов

Для воспроизведения экспериментов используйте DVC pipeline:

```bash
# Запустить весь пайплайн (prepare -> train -> evaluate)
dvc repro

# Запустить конкретный stage
dvc repro prepare
dvc repro train
dvc repro evaluate

# Посмотреть метрики
dvc metrics show

# Сравнить метрики между экспериментами
dvc metrics diff
```

### Обучение модели

#### Через DVC Pipeline (рекомендуется)

```bash
# Полный пайплайн: подготовка данных -> обучение -> оценка
dvc repro

# Только обучение (если данные уже подготовлены)
dvc repro train
```

#### Напрямую через Python

```bash
# Базовый запуск
python src/training/train.py --config configs/config.yaml

# С подробным логированием
python src/training/train.py --config configs/config.yaml --verbose
```

### Параметры скрипта обучения

- `--config` (обязательный): Путь к конфигурационному файлу YAML
- `--verbose`: Включить подробный режим логирования (DEBUG уровень)

### Конфигурация

Основные параметры настраиваются в `configs/config.yaml`:

- **Данные**: путь к данным, batch size, количество воркеров
- **Модель**: архитектура, количество классов, размеры скрытых слоев
- **Обучение**: количество эпох, learning rate, оптимизатор, scheduler
- **Логирование**: директория для логов

### Результаты обучения

После обучения будут созданы:

- `models/best_model/` - лучшая модель (по валидационной точности) в формате Hugging Face
- `models/final_model/` - финальная модель после всех эпох в формате Hugging Face
- `logs/training.log` - лог файл с детальной информацией о процессе обучения
- `metrics/test_metrics.json` - метрики оценки модели на тестовой выборке
- `metrics/data_info.yaml` - информация о датасете

Модели сохраняются в формате Hugging Face (с `config.json` и `model.safetensors`).

### Оценка модели

```bash
# Через DVC Pipeline
dvc repro evaluate

# Напрямую через Python
python src/training/evaluate.py --config configs/config.yaml --model-path models/best_model
```

## Работа с MLflow

### Трекинг экспериментов

Все эксперименты автоматически логируются в MLflow при запуске обучения. MLflow отслеживает:

- **Параметры**: learning rate, batch size, optimizer, epochs, и др.
- **Метрики**: train/val/test loss, accuracy, precision, recall, F1-score
- **Артефакты**: модели, логи, конфигурации, dvc.lock
- **DVC интеграция**: хеш данных из data.dvc логируется как тег

### Просмотр результатов экспериментов

```bash
# Запустить MLflow UI
mlflow ui

# MLflow UI будет доступен по адресу: http://localhost:5000
```

В интерфейсе MLflow вы можете:
- Просматривать все запуски (runs) с параметрами и метриками
- Сравнивать эксперименты
- Визуализировать метрики (графики обучения)
- Скачивать артефакты (модели, логи)
- Фильтровать и сортировать эксперименты

### Где хранятся результаты MLflow

Результаты экспериментов хранятся локально в директории `mlruns/`:
- `mlruns/` - база данных экспериментов и метрик
- `mlartifacts/` - артефакты (модели, файлы)

Эти директории добавлены в `.gitignore` и не версионируются в git.

### Интеграция DVC + MLflow

При каждом запуске обучения:
1. MLflow логирует хеш данных из `data.dvc` как тег `dvc_data_hash`
2. Файл `dvc.lock` сохраняется как артефакт
3. Это позволяет точно знать, на какой версии данных обучалась модель

### Сравнение экспериментов

```bash
# В MLflow UI можно:
# 1. Выбрать несколько runs (checkbox)
# 2. Нажать "Compare"
# 3. Увидеть разницу в параметрах и метриках
```

### Программный доступ к MLflow

```python
import mlflow

# Получить лучший run по метрике
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("fashion-mnist-classification")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_accuracy DESC"],
    max_results=1
)
best_run = runs[0]
print(f"Best run ID: {best_run.info.run_id}")
print(f"Test accuracy: {best_run.data.metrics['test_accuracy']}")
```

## Docker контейнеризация

Проект упакован в Docker образ для offline inference (пакетной обработки изображений).

### Что делает скрипт `src/predict.py`

Скрипт для пакетного инференса:
- **Загружает модель** с диска (`models/best_model/`)
- **Принимает аргументы**:
  - `--input_path` - путь к директории с изображениями
  - `--output_path` - путь для сохранения результатов (CSV)
  - `--model_path` - путь к модели (по умолчанию `models/best_model`)
  - `--batch_size` - размер батча для инференса (по умолчанию 32)
  - `--device` - устройство (cpu/cuda)
- **Обрабатывает изображения**: загружает, изменяет размер, нормализует
- **Выполняет предсказания** на всех изображениях
- **Сохраняет результаты** в CSV файл с колонками:
  - `filename` - имя файла
  - `predicted_class` - предсказанный класс (0-9)
  - `predicted_label` - название класса (T-shirt, Dress, и т.д.)
  - `confidence` - уверенность модели
  - `prob_<class_name>` - вероятности для каждого класса

### Форматы данных

**Входные данные** (`--input_path`):
- Директория с изображениями
- Поддерживаемые форматы: `.png`, `.jpg`, `.jpeg`, `.bmp`
- Изображения автоматически конвертируются в grayscale и resize до 28x28

**Выходные данные** (`--output_path`):
- CSV файл с предсказаниями
- Пример:
```csv
filename,predicted_class,predicted_label,confidence,prob_T-shirt/top,prob_Trouser,...
image1.png,0,T-shirt/top,0.95,0.95,0.02,...
image2.jpg,3,Dress,0.88,0.05,0.01,...
```

### Сборка Docker образа

```bash
# Сборка образа
docker build -t ml-app:v1 .

# Проверка что образ создан
docker images | grep ml-app
```

### Запуск контейнера

#### Базовый запуск

```bash
# Подготовка директорий
mkdir -p test_images predictions

# Запуск контейнера
docker run --rm \
    -v $(pwd)/test_images:/app/input:ro \
    -v $(pwd)/predictions:/app/output \
    ml-app:v1 \
    --input_path /app/input \
    --output_path /app/output/predictions.csv \
    --model_path /app/models/best_model \
    --batch_size 32 \
    --device cpu
```

#### Использование готового скрипта

```bash
# Запуск через готовый скрипт
./docker-run-example.sh
```

#### Пояснение параметров Docker

- `--rm` - автоматически удалить контейнер после завершения
- `-v $(pwd)/test_images:/app/input:ro` - монтировать директорию с изображениями (read-only)
- `-v $(pwd)/predictions:/app/output` - монтировать директорию для результатов
- `ml-app:v1` - имя образа
- Далее идут аргументы для `src/predict.py`

### Структура Docker образа

```
/app/
├── src/                    # Исходный код
│   ├── predict.py         # Скрипт инференса (ENTRYPOINT)
│   ├── models/            # Архитектуры моделей
│   └── ...
├── models/                # Сохранённые модели
│   └── best_model/
├── configs/               # Конфигурации
└── requirements.txt       # Зависимости
```

### Что игнорируется (.dockerignore)

Для уменьшения размера образа исключены:
- Данные обучения (`data/`)
- MLflow артефакты (`mlruns/`, `mlartifacts/`)
- DVC кеш (`.dvc/cache/`)
- Логи, тесты, документация
- Временные файлы Python

### Размер образа

- **Базовый образ**: `python:3.9-slim` (~150 MB)
- **С зависимостями**: ~1-2 GB (PyTorch, numpy, pandas)
- **С моделью**: +2-10 MB (в зависимости от модели)

### Оптимизация образа (опционально)

Для production можно:
1. Использовать multi-stage build
2. Не копировать модель в образ, а монтировать через volume
3. Использовать более легкий базовый образ
4. Квантизовать модель

### Пример использования

```bash
# 1. Положить изображения в test_images/
cp /path/to/images/*.png test_images/

# 2. Запустить контейнер
docker run --rm \
    -v $(pwd)/test_images:/app/input:ro \
    -v $(pwd)/predictions:/app/output \
    ml-app:v1 \
    --input_path /app/input \
    --output_path /app/output/predictions.csv

# 3. Проверить результаты
cat predictions/predictions.csv
```

### Интеграция с DVC

Если модель версионируется через DVC и не включена в образ:

```bash
# Получить модель через DVC
dvc pull models/best_model

# Запустить контейнер с монтированием модели
docker run --rm \
    -v $(pwd)/test_images:/app/input:ro \
    -v $(pwd)/predictions:/app/output \
    -v $(pwd)/models:/app/models:ro \
    ml-app:v1 \
    --input_path /app/input \
    --output_path /app/output/predictions.csv
```

## TorchServe - онлайн REST API сервис

Модель развернута как REST API сервис с использованием TorchServe для online inference.

### Архитектура решения

```
Client → REST API (8080) → TorchServe → Handler → Model → Response
         ↓
    Management API (8081)
         ↓  
    Metrics API (8082)
```

### Подготовка артефактов

#### 1. Экспорт модели в TorchServe формат

```bash
# Экспорт модели в state_dict формат
python export_model_for_torchserve.py \
    --model-path models/best_model \
    --output-path torchserve/model.pt
```

Создаст:
- `torchserve/model.pt` - веса модели
- `torchserve/model_config.json` - конфигурация

#### 2. Создание MAR архива

```bash
# Создание model archive с помощью torch-model-archiver
./create_mar_archive.sh
```

Создаст: `torchserve/model-store/fashion_mnist.mar`

MAR архив содержит:
- Веса модели (`model.pt`)
- Handler (`handler.py`) с предобработкой и постобработкой
- Метаданные модели

### Handler - пред- и постобработка

Handler (`torchserve/handler.py`) реализует:

**Предобработка (`preprocess`):**
- Декодирование base64 изображения
- Конвертация в PIL Image
- Resize до 28x28, grayscale
- Нормализация
- Преобразование в тензор

**Постобработка (`postprocess`):**
- Вычисление вероятностей (softmax)
- Определение предсказанного класса
- Формирование JSON ответа с:
  - `predicted_class` - номер класса (0-9)
  - `predicted_label` - название класса
  - `confidence` - уверенность модели
  - `probabilities` - вероятности для всех классов

### Сборка и запуск TorchServe

#### Автоматический запуск (рекомендуется)

```bash
# Запуск всего процесса одной командой
./run_torchserve.sh
```

Скрипт автоматически:
1. Экспортирует модель
2. Создаст MAR архив
3. Соберет Docker образ
4. Запустит контейнер
5. Проверит здоровье сервиса

#### Ручной запуск

```bash
# 1. Экспорт модели
python export_model_for_torchserve.py

# 2. Создание MAR архива
./create_mar_archive.sh

# 3. Сборка образа
cd torchserve
docker build -t mymodel-serve:v1 .

# 4. Запуск контейнера
docker run -d \
    -p 8080:8080 \
    -p 8081:8081 \
    -p 8082:8082 \
    --name torchserve-container \
    mymodel-serve:v1
```

### Проверка работы сервиса

#### Health check

```bash
# Проверка что сервис запущен
curl http://localhost:8080/ping

# Ожидаемый ответ:
# {"status": "Healthy"}
```

#### Список моделей

```bash
# Получить список зарегистрированных моделей
curl http://localhost:8081/models

# Ответ:
# {
#   "models": [
#     {
#       "modelName": "fashion_mnist",
#       "modelUrl": "fashion_mnist.mar"
#     }
#   ]
# }
```

### REST API - примеры запросов

#### Формат входных данных

TorchServe принимает изображения в формате:

**1. JSON с base64:**
```json
{
  "image": "base64_encoded_image_data..."
}
```

**2. Binary (multipart/form-data):**
```bash
curl -X POST \
  http://localhost:8080/predictions/fashion_mnist \
  -F "data=@image.png"
```

#### Пример запроса

```bash
# С тестовым файлом
curl -X POST \
  http://localhost:8080/predictions/fashion_mnist \
  -T torchserve/sample_input.json \
  -H "Content-Type: application/json"
```

#### Формат ответа

```json
{
  "predicted_class": 3,
  "predicted_label": "Dress",
  "confidence": 0.9234,
  "probabilities": {
    "T-shirt/top": 0.0123,
    "Trouser": 0.0045,
    "Pullover": 0.0234,
    "Dress": 0.9234,
    "Coat": 0.0156,
    "Sandal": 0.0034,
    "Shirt": 0.0089,
    "Sneaker": 0.0023,
    "Bag": 0.0045,
    "Ankle boot": 0.0017
  }
}
```

### Management API

```bash
# Информация о конкретной модели
curl http://localhost:8081/models/fashion_mnist

# Удаление модели (unregister)
curl -X DELETE http://localhost:8081/models/fashion_mnist

# Регистрация новой версии
curl -X POST "http://localhost:8081/models?url=fashion_mnist.mar"

# Масштабирование (увеличение workers)
curl -X PUT "http://localhost:8081/models/fashion_mnist?min_worker=4"
```

### Metrics API

```bash
# Получить метрики
curl http://localhost:8082/metrics

# Метрики включают:
# - Количество запросов
# - Время обработки (latency)
# - Использование CPU/Memory
# - Ошибки
```

### Параметры конфигурации

Файл `torchserve/config.properties`:

```properties
# Количество workers на модель (параллельная обработка)
default_workers_per_model=2

# Размер очереди запросов
job_queue_size=100

# Максимальный размер запроса/ответа
max_request_size=655350000
max_response_size=655350000

# Порты
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
```

### Управление контейнером

```bash
# Просмотр логов
docker logs torchserve-container

# Просмотр логов в реальном времени
docker logs -f torchserve-container

# Остановка сервиса
docker stop torchserve-container

# Запуск остановленного контейнера
docker start torchserve-container

# Удаление контейнера
docker rm torchserve-container
```

### Производительность

- **Latency**: ~50-100ms на запрос (зависит от hardware)
- **Throughput**: ~20-50 requests/sec (на 1 worker)
- **Масштабирование**: Увеличение `default_workers_per_model` для параллельной обработки

### Интеграция с DVC

Модель версионируется через DVC, поэтому:

1. **Для обновления модели:**
```bash
# Получить новую версию модели через DVC
dvc pull models/best_model

# Пересоздать MAR архив
./create_mar_archive.sh

# Пересобрать Docker образ
cd torchserve && docker build -t mymodel-serve:v1 .

# Перезапустить контейнер
docker restart torchserve-container
```

2. **Связь версий:** В MAR архив можно добавить `dvc.lock` для отслеживания версии данных

### Структура TorchServe проекта

```
torchserve/
├── Dockerfile              # Docker образ с TorchServe
├── config.properties       # Конфигурация сервиса
├── handler.py             # Custom handler с пред-/постобработкой
├── model-store/           # Директория с MAR архивами
│   └── fashion_mnist.mar  # Model archive
└── sample_input.json      # Пример входных данных

export_model_for_torchserve.py  # Экспорт модели
create_mar_archive.sh            # Создание MAR
run_torchserve.sh                # Автоматический запуск
```

## Тестирование

Проект включает комплексное тестирование всех компонентов:

### Запуск тестов

```bash
# Все тесты
pytest tests/ -v

# С покрытием кода
pytest tests/ -v --cov=src --cov-report=html

# Конкретный тест
pytest tests/test_data_validation.py -v
```

### Структура тестов

- `tests/test_data_validation.py` - Тесты валидации данных (формат, типы, признаки, метки)
- `tests/test_prediction_conversion.py` - Тесты конвертации предсказаний модели
- `tests/test_training_pipeline.py` - Тесты pipeline обучения (утилиты, эпохи, воспроизводимость)

### Покрытие тестами

Тесты проверяют:
- Корректность предобработки данных
- Работу pipeline обучения
- Конвертацию предсказаний в финальные значения
- Валидацию данных (формат, типы, диапазоны)
- Воспроизводимость результатов

**Важно:** Тесты проверяют корректность работы pipeline, а не качество модели.

## CI/CD

Проект настроен с автоматическим CI/CD через GitHub Actions:

### Автоматические проверки

При каждом коммите автоматически выполняются:
- Запуск всех тестов на Python 3.9, 3.10, 3.11
- Проверка покрытия кода тестами
- Проверка форматирования кода (black)
- Линтинг кода (flake8)

### GitHub Actions Workflow

Файл `.github/workflows/ci.yml` содержит:
- Job `test`: Запуск тестов на разных версиях Python
- Job `lint`: Проверка форматирования и линтинга

### Локальная проверка

```bash
# Проверка форматирования
black --check src/ tests/

# Форматирование кода
black src/ tests/

# Линтинг
flake8 src/ tests/
```

## Воспроизводимость

Для обеспечения воспроизводимости результатов:

1. Установлен `random_seed` в конфигурации (по умолчанию 42)
2. Все случайные операции используют этот seed
3. Конфигурация сохраняется вместе с моделью
4. Логи содержат все параметры обучения
5. Тесты проверяют воспроизводимость результатов
6. **DVC** обеспечивает версионирование данных, моделей и пайплайнов
7. **DVC pipeline** позволяет полностью восстановить любую версию датасета, модели и пайплайна одной командой `dvc pull`

### Восстановление конкретной версии эксперимента

```bash
# Переключиться на нужный коммит
git checkout <commit-hash>

# Получить соответствующие данные и модели
dvc checkout
dvc pull

# Воспроизвести эксперимент
dvc repro
```
