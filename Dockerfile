FROM python:3.10-slim

WORKDIR /app

# Устанавливаем только самое необходимое
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем только requirements для кеширования
COPY requirements.txt .

# Устанавливаем зависимости без кеша
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы
COPY . .

# Чистим ненужное
RUN find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Запускаем
CMD gunicorn app:app --bind 0.0.0.0:$PORT