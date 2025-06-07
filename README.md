# 🛰️ EuroSAT Классификатор

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.21+-red.svg)](https://streamlit.io/)

Веб-приложение для классификации спутниковых снимков на основе датасета EuroSAT с использованием глубокого обучения.

## 📋 Описание

Проект представляет собой полноценное решение для автоматической классификации типов земного покрова по спутниковым снимкам. Приложение поддерживает работу с мультиспектральными изображениями и различными форматами геоданных.

### 🎯 Основные возможности

- **Классификация 10 типов земного покрова**: леса, реки, дороги, жилые районы и др.
- **Поддержка мультиспектральных снимков**: работа с многоканальными GeoTIFF
- **Веб-интерфейс**: удобный интерфейс на Streamlit
- **REST API**: FastAPI для интеграции с другими системами
- **Экспорт результатов**: PNG, GeoTIFF, GeoJSON

## 🏗️ Архитектура

```
┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
│   Streamlit UI  │────▶│  FastAPI     │────▶│  TensorFlow    │
│   (Frontend)    │     │  (Backend)   │     │  (Model)       │
└─────────────────┘     └──────────────┘     └────────────────┘
```

## 📦 Установка

### Требования
- Python 3.8+
- GDAL библиотеки (для работы с геоданными)

### Шаги установки

1. **Клонируйте репозиторий**
```bash
git clone https://github.com/yourusername/eurosat-classifier.git
cd eurosat-classifier
```

2. **Создайте виртуальное окружение**
```bash
# Для conda (рекомендуется)
conda create -n eurosat python=3.9
conda activate eurosat

# Или используйте venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

3. **Установите зависимости**
```bash
# Установка GDAL (для conda)
conda install -c conda-forge gdal

# Установка Python пакетов
pip install -r requirements.txt
```

4. **Скачайте обученную модель**
```bash
# Создайте папку для модели
mkdir models
# Скачайте модель (ссылка будет предоставлена)
```

## 🚀 Запуск

### Запуск backend (FastAPI)
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Запуск frontend (Streamlit)
```bash
streamlit run app.py
```

Приложение будет доступно по адресу: http://localhost:8501

## 💻 Использование

### Через веб-интерфейс

1. Откройте браузер и перейдите на http://localhost:8501
2. Загрузите спутниковый снимок (поддерживаются форматы: GeoTIFF, TIFF, JPG, PNG)
3. Для мультиспектральных снимков выберите каналы RGB
4. Нажмите "Классифицировать"
5. Скачайте результаты в нужном формате

### Через API

```python
import requests

# Проверка состояния
response = requests.get("http://localhost:8000/health")
print(response.json())

# Классификация изображения
with open("image.tif", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/classify_all/", files=files)
    results = response.json()
```

## 📊 Классы земного покрова

| Класс | Название | Цвет | Описание |
|-------|----------|------|----------|
| 0 | AnnualCrop | 🟡 Жёлтый | Однолетние культуры |
| 1 | Forest | 🟢 Тёмно-зелёный | Лес |
| 2 | HerbaceousVegetation | 🟢 Светло-зелёный | Травянистая растительность |
| 3 | Highway | 🔘 Серый | Автомагистрали |
| 4 | Industrial | 🟤 Коричневый | Промышленные зоны |
| 5 | Pasture | 🟢 Бледно-зелёный | Пастбища |
| 6 | PermanentCrop | 🟠 Оранжевый | Многолетние культуры |
| 7 | Residential | 🔴 Красный | Жилые районы |
| 8 | River | 🔵 Синий | Реки |
| 9 | SeaLake | 🔵 Голубой | Моря и озёра |

## 📁 Структура проекта

```
eurosat-classifier/
├── app.py              # Streamlit интерфейс
├── main.py             # FastAPI сервер
├── classifier.py       # Основная логика классификации
├── requirements.txt    # Зависимости Python
├── models/            
│   └── best_eurosat_model.h5  # Обученная модель
├── README.md          
└── examples/          # Примеры изображений
```

## 🔧 Конфигурация

### Параметры классификатора

```python
# В classifier.py
patch_size = 64      # Размер патча для классификации
batch_size = 64      # Размер батча для предсказаний
```

### Настройка для Google Colab

Для работы в Google Colab используйте следующий код:

```python
# Установка зависимостей
!pip install -r requirements.txt

# Запуск с использованием ngrok для публичного доступа
!pip install pyngrok
from pyngrok import ngrok

# Запуск FastAPI
!uvicorn main:app --port 8000 &

# Создание туннеля
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")
```

## 📈 Производительность

- **Скорость обработки**: ~2-5 сек для изображения 512x512 пикселей
- **Точность модели**: 94% на тестовой выборке EuroSAT


