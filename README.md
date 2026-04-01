# 🍄 Mushroom ML Dashboard

Система интеллектуального анализа данных для классификации грибов (съедобные/ядовитые) на базе [UCI Mushroom Dataset](https://archive.ics.uci.edu/dataset/73/mushroom).

## 📋 Возможности

- **EDA** — разведочный анализ данных с визуализацией
- **Классификация** — Naive Bayes vs Random Forest с метриками (Accuracy, Precision, Recall, F1, ROC-AUC)
- **Кластеризация** — k-means (с подбором k по Silhouette Score) и DBSCAN
- **Ассоциативные правила** — Apriori (support ≥ 0.3, confidence ≥ 0.8)
- **Поиск аномалий** — Isolation Forest и LOF
- **Web Dashboard** — интерактивный React фронтенд с анимациями
- **Prediction API** — REST endpoint для предсказания по признакам гриба
- **Тесты** — полный набор pytest-тестов

## 🗂 Структура проекта

```
ml_with_mushrooms/
├── mushroom_data.csv          # Датасет UCI Mushroom
├── main.py                    # ML-пайплайн (EDA, обучение, оценка)
├── api.py                     # FastAPI сервер (WebSocket + REST API)
├── run_dashboard.py           # Скрипт запуска бэкенда
├── requirements.txt           # Python зависимости
│
├── ml/                        # ML модуль
│   ├── train_pipeline.py      # Полный пайплайн: обучение + кластеризация + Apriori + аномалии
│   ├── models/
│   │   └── mushroom_rf_model.pkl  # Обученная модель Random Forest
│   └── plots/                 # Графики пайплайна
│       ├── feature_importances.png
│       ├── silhouette_kmeans.png
│       ├── kmeans_pca.png
│       ├── dbscan_pca.png
│       ├── apriori_rules.png
│       ├── anomaly_detection.png
│       └── association_rules_top20.csv
│
├── frontend/                  # React (Vite) Dashboard
│   ├── src/
│   │   ├── App.jsx            # Главный компонент (Dashboard + Prediction)
│   │   ├── App.css            # Стили компонентов
│   │   ├── index.css          # Глобальные стили (дизайн-система)
│   │   └── main.jsx           # Точка входа
│   └── package.json
│
└── tests/                     # Тесты (pytest)
    └── test_pipeline.py       # 17 тестов: данные, модели, API
```

## 🚀 Запуск

### 1. Установка зависимостей

```bash
# Python
pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

### 2. Обучение модели (если .pkl отсутствует)

```bash
python ml/train_pipeline.py
```

### 3. Запуск Dashboard

```bash
# Бэкенд (терминал 1)
python run_dashboard.py
# → API: http://localhost:8000

# Фронтенд (терминал 2)
cd frontend && npm run dev
# → UI: http://localhost:5173
```

### 4. Запуск тестов

```bash
pytest tests/ -v
```

## 🔌 API Endpoints

| Метод | URL               | Описание                                      |
|-------|-------------------|-----------------------------------------------|
| WS    | `/ws/run`         | WebSocket — стрим прогресса ML пайплайна       |
| GET   | `/api/features`   | Описание всех признаков и вариантов            |
| POST  | `/api/predict`    | Предсказание: съедобный или ядовитый           |
| GET   | `/api/model-info` | Метрики модели и top-features                  |

### Пример запроса `/api/predict`

```json
POST /api/predict
{
  "features": {
    "odor": "n",
    "spore_print_color": "k",
    "gill_color": "n",
    "ring_type": "p",
    "stalk_root": "b"
  }
}
```

## ⚠️ Дисклеймер

> **Бот создан исключительно в учебных целях!**
> Ни в коем случае не используйте прогнозы модели для реального сбора грибов.
> Всегда консультируйтесь с опытным микологом!
