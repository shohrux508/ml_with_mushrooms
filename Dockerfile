# Использование официального образа Python 3.12 (slim для уменьшения размера)
FROM python:3.12-slim

# Установка системных зависимостей:
# - curl для установки Node.js
# - gcc и python3-dev для сборки некоторых Python библиотек (например, scikit-learn)
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    python3-dev \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Сначала копируем зависимости фронтенда и устанавливаем их
# Это позволяет кешировать слои Docker, если package.json не менялся
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm install

# 2. Копируем исходники фронтенда и собираем его
COPY frontend ./frontend
RUN cd frontend && npm run build

# 3. Установка зависимостей Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Копируем остальные файлы проекта (включая api.py, ml/, mushroom_data.csv)
COPY . .

# 5. Запуск пайплайна обучения
# Это создаст файлы моделей и графики, которые нужны API
RUN python ml/train_pipeline.py

# 6. Настройка переменных окружения
# Railway автоматически подставляет порт в переменную $PORT
ENV PORT 8000

# Команда для запуска приложения через uvicorn
# Используем uvicorn как модуль через python -m
CMD ["sh", "-c", "python -m uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
