# Dockerfile для TorchServe с Fashion-MNIST моделью
FROM pytorch/torchserve:latest

# Установка рабочей директории
WORKDIR /home/model-server

# Копирование конфигурации TorchServe
COPY config.properties ./config.properties

# Копирование MAR архива модели
COPY model-store/fashion_mnist.mar ./model-store/

# Expose порты
# 8080 - Inference API
# 8081 - Management API  
# 8082 - Metrics API
EXPOSE 8080 8081 8082

# Запуск TorchServe с автоматической регистрацией модели
CMD ["torchserve", \
     "--start", \
     "--ts-config", "/home/model-server/config.properties", \
     "--models", "fashion_mnist=fashion_mnist.mar", \
     "--model-store", "/home/model-server/model-store"]

