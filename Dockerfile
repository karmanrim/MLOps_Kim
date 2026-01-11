# Используем официальный Python образ
FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование requirements.txt и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода проекта
COPY src/ ./src/
COPY configs/ ./configs/

# Копирование сохранённой модели (или подтянуть через DVC)
COPY models/ ./models/

# Установка переменных окружения
ENV PYTHONUNBUFFERED=1

# Точка входа - скрипт для инференса
ENTRYPOINT ["python", "-m", "src.predict"]

# Аргументы по умолчанию (можно переопределить при запуске)
CMD ["--help"]

