# Imagen base estable y compatible con mediapipe
FROM python:3.10-slim

# Variables para evitar problemas comunes
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema necesarias para:
# - OpenCV
# - Mediapipe
# - FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar primero requirements para cache eficiente
COPY requirements.txt .

# Actualizar pip e instalar dependencias
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Comando de arranque compatible con Render (usa variable PORT)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]