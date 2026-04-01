"""
api.py — FastAPI сервер для ML Dashboard + Prediction API.
Импортирует логику из main.py, НЕ смешивая её с веб-логикой.
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Mushroom ML Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Папка проекта (где лежит main.py и сохраняются графики)
PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "ml" / "models" / "mushroom_rf_model.pkl"

# Раздаём статику (графики PNG) по /plots/...
app.mount("/plots", StaticFiles(directory=str(PROJECT_DIR)), name="plots")

# Раздаём графики из ml/plots
ML_PLOTS_DIR = PROJECT_DIR / "ml" / "plots"
if ML_PLOTS_DIR.exists():
    app.mount("/ml-plots", StaticFiles(directory=str(ML_PLOTS_DIR)), name="ml_plots")

# Раздаём собранный фронтенд (production)
FRONTEND_DIST = PROJECT_DIR / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="frontend_assets")

# ── Глобальный кеш модели ──────────────────────────────────────────────────
_model_cache: Optional[dict] = None
_modes_cache: Optional[dict] = None


def get_model():
    """Загружает модель из .pkl файла (с кешированием)."""
    global _model_cache
    if _model_cache is None:
        if not MODEL_PATH.exists():
            return None
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def get_modes(feature_columns):
    """Возвращает модальные значения признаков (с кешированием)."""
    global _modes_cache
    if _modes_cache is None:
        df = pd.read_csv(str(PROJECT_DIR / "mushroom_data.csv"))
        if "veil_type" in df.columns:
            df = df.drop(columns=["veil_type"])
        df["stalk_root"] = df["stalk_root"].replace("?", "Unknown")
        _modes_cache = {col: df[col].mode()[0] for col in feature_columns}
    return _modes_cache


# ── Маппинг признаков и значений для интерфейса ────────────────────────────
# Используются все 22 признака из UCI Mushroom Dataset (без veil_type)
FEATURE_OPTIONS = {
    "cap_shape": {
        "label": "Форма шляпки",
        "options": {
            "b": "Колокольчатая (bell)",
            "c": "Коническая (conical)",
            "x": "Выпуклая (convex)",
            "f": "Плоская (flat)",
            "k": "Бугорчатая (knobbed)",
            "s": "Вдавленная (sunken)",
        },
    },
    "cap_surface": {
        "label": "Поверхность шляпки",
        "options": {
            "f": "Волокнистая (fibrous)",
            "g": "Рифлёная (grooves)",
            "y": "Чешуйчатая (scaly)",
            "s": "Гладкая (smooth)",
        },
    },
    "cap_color": {
        "label": "Цвет шляпки",
        "options": {
            "n": "Коричневый (brown)",
            "b": "Жёлто-коричневый (buff)",
            "c": "Коричный (cinnamon)",
            "g": "Серый (gray)",
            "r": "Зелёный (green)",
            "p": "Розовый (pink)",
            "u": "Фиолетовый (purple)",
            "e": "Красный (red)",
            "w": "Белый (white)",
            "y": "Жёлтый (yellow)",
        },
    },
    "bruises": {
        "label": "Синяки",
        "options": {"t": "Да (bruises)", "f": "Нет (no)"},
    },
    "odor": {
        "label": "Запах",
        "options": {
            "a": "Миндальный (almond)",
            "l": "Анисовый (anise)",
            "c": "Креозотный (creosote)",
            "y": "Рыбный (fishy)",
            "f": "Гнилостный (foul)",
            "m": "Затхлый (musty)",
            "n": "Без запаха (none)",
            "p": "Едкий (pungent)",
            "s": "Пряный (spicy)",
        },
    },
    "gill_attachment": {
        "label": "Крепление пластинок",
        "options": {
            "a": "Прикреплённые (attached)",
            "d": "Нисходящие (descending)",
            "f": "Свободные (free)",
            "n": "Зубчатые (notched)",
        },
    },
    "gill_spacing": {
        "label": "Расстояние пластинок",
        "options": {
            "c": "Плотные (close)",
            "w": "Широкие (crowded)",
            "d": "Далёкие (distant)",
        },
    },
    "gill_size": {
        "label": "Размер пластинок",
        "options": {"b": "Широкие (broad)", "n": "Узкие (narrow)"},
    },
    "gill_color": {
        "label": "Цвет пластинок",
        "options": {
            "k": "Чёрный (black)",
            "n": "Коричневый (brown)",
            "b": "Жёлто-коричневый (buff)",
            "h": "Шоколадный (chocolate)",
            "g": "Серый (gray)",
            "r": "Зелёный (green)",
            "o": "Оранжевый (orange)",
            "p": "Розовый (pink)",
            "u": "Фиолетовый (purple)",
            "e": "Красный (red)",
            "w": "Белый (white)",
            "y": "Жёлтый (yellow)",
        },
    },
    "stalkshape": {
        "label": "Форма ножки",
        "options": {"e": "Расширяющаяся (enlarging)", "t": "Сужающаяся (tapering)"},
    },
    "stalk_root": {
        "label": "Корень ножки",
        "options": {
            "b": "Луковичный (bulbous)",
            "c": "Булавовидный (club)",
            "u": "Кубковидный (cup)",
            "e": "Равномерный (equal)",
            "z": "Корневищный (rhizomorphs)",
            "r": "Укоренённый (rooted)",
            "?": "Неизвестно (missing)",
        },
    },
    "stalk_surface_above_ring": {
        "label": "Поверхность ножки выше кольца",
        "options": {
            "f": "Волокнистая (fibrous)",
            "y": "Чешуйчатая (scaly)",
            "k": "Шёлковая (silky)",
            "s": "Гладкая (smooth)",
        },
    },
    "stalk_surface_below_ring": {
        "label": "Поверхность ножки ниже кольца",
        "options": {
            "f": "Волокнистая (fibrous)",
            "y": "Чешуйчатая (scaly)",
            "k": "Шёлковая (silky)",
            "s": "Гладкая (smooth)",
        },
    },
    "stalk_color_above_ring": {
        "label": "Цвет ножки выше кольца",
        "options": {
            "n": "Коричневый (brown)",
            "b": "Жёлто-коричневый (buff)",
            "c": "Коричный (cinnamon)",
            "g": "Серый (gray)",
            "o": "Оранжевый (orange)",
            "p": "Розовый (pink)",
            "e": "Красный (red)",
            "w": "Белый (white)",
            "y": "Жёлтый (yellow)",
        },
    },
    "stalk_color_below_ring": {
        "label": "Цвет ножки ниже кольца",
        "options": {
            "n": "Коричневый (brown)",
            "b": "Жёлто-коричневый (buff)",
            "c": "Коричный (cinnamon)",
            "g": "Серый (gray)",
            "o": "Оранжевый (orange)",
            "p": "Розовый (pink)",
            "e": "Красный (red)",
            "w": "Белый (white)",
            "y": "Жёлтый (yellow)",
        },
    },
    "veil_color": {
        "label": "Цвет покрывала",
        "options": {
            "n": "Коричневый (brown)",
            "o": "Оранжевый (orange)",
            "w": "Белый (white)",
            "y": "Жёлтый (yellow)",
        },
    },
    "ring_number": {
        "label": "Количество колец",
        "options": {
            "n": "Нет (none)",
            "o": "Одно (one)",
            "t": "Два (two)",
        },
    },
    "ring_type": {
        "label": "Тип кольца",
        "options": {
            "c": "Паутинное (cobwebby)",
            "e": "Быстротечное (evanescent)",
            "f": "Расширяющееся (flaring)",
            "l": "Большое (large)",
            "n": "Нет (none)",
            "p": "Висячее (pendant)",
            "s": "Оболочечное (sheathing)",
            "z": "Зональное (zone)",
        },
    },
    "spore_print_color": {
        "label": "Цвет спорового отпечатка",
        "options": {
            "k": "Чёрный (black)",
            "n": "Коричневый (brown)",
            "b": "Жёлто-коричневый (buff)",
            "h": "Шоколадный (chocolate)",
            "r": "Зелёный (green)",
            "o": "Оранжевый (orange)",
            "u": "Фиолетовый (purple)",
            "w": "Белый (white)",
            "y": "Жёлтый (yellow)",
        },
    },
    "population": {
        "label": "Численность",
        "options": {
            "a": "Обильная (abundant)",
            "c": "Кластерная (clustered)",
            "n": "Многочисленная (numerous)",
            "s": "Рассеянная (scattered)",
            "v": "Несколько (several)",
            "y": "Одиночная (solitary)",
        },
    },
    "habitat": {
        "label": "Среда обитания",
        "options": {
            "g": "Трава (grasses)",
            "l": "Листья (leaves)",
            "m": "Луга (meadows)",
            "p": "Тропинки (paths)",
            "u": "Городская (urban)",
            "w": "Отходы (waste)",
            "d": "Леса (woods)",
        },
    },
}

# Топ-5 самых важных признаков (определены по feature importance)
TOP_FEATURES = ["odor", "spore_print_color", "gill_color", "ring_type", "stalk_root"]


# ── Pydantic-модели ───────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    """Запрос предсказания. Достаточно передать top‑5 признаков."""
    features: Dict[str, str]


class PredictionResponse(BaseModel):
    prediction: str  # "Съедобный" | "Ядовитый"
    probability: float
    disclaimer: str


# ── REST Endpoints ──────────────────────────────────────────────────────────
@app.get("/api/features")
async def get_features():
    """Возвращает описание всех признаков и их вариантов."""
    return {
        "all_features": FEATURE_OPTIONS,
        "top_features": TOP_FEATURES,
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    """Предсказание на основе введённых признаков."""
    artifact = get_model()
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Модель ещё не обучена. Запустите ml/train_pipeline.py",
        )

    model = artifact["model"]
    ohe_columns = artifact["ohe_columns"]
    feature_columns = artifact["feature_columns"]

    # Заполним все признаки модальным значением, потом перезапишем указанные
    modes = get_modes(feature_columns)

    # Создаём строку данных
    row = dict(modes)
    for feat, val in req.features.items():
        if feat in row:
            if feat == "stalk_root" and val == "?":
                row[feat] = "Unknown"
            else:
                row[feat] = val

    # One-Hot Encoding
    row_df = pd.DataFrame([row])
    row_ohe = pd.get_dummies(row_df)

    # Выравниваем столбцы — создаём полный нулевой вектор и заполняем
    missing_cols = [col for col in ohe_columns if col not in row_ohe.columns]
    if missing_cols:
        zeros = pd.DataFrame(0, index=row_ohe.index, columns=missing_cols)
        row_ohe = pd.concat([row_ohe, zeros], axis=1)
    row_ohe = row_ohe[ohe_columns]

    # Предсказание
    pred = model.predict(row_ohe)[0]
    prob = model.predict_proba(row_ohe)[0]

    result = "Съедобный 🟢" if pred == 0 else "Ядовитый 🔴"
    confidence = prob[pred]

    return PredictionResponse(
        prediction=result,
        probability=round(float(confidence), 4),
        disclaimer=(
            "⚠️ ДИСКЛЕЙМЕР: Бот создан в учебных целях! "
            "Ни в коем случае не используйте этот прогноз для реального сбора грибов. "
            "Всегда консультируйтесь с опытным микологом!"
        ),
    )


@app.get("/api/model-info")
async def model_info():
    """Информация о модели."""
    artifact = get_model()
    if artifact is None:
        raise HTTPException(status_code=503, detail="Модель не обучена")

    return {
        "metrics": artifact["metrics"],
        "top_features": artifact["top_features"],
        "n_features": len(artifact["ohe_columns"]),
    }


# ── WebSocket helpers ──────────────────────────────────────────────────────
async def send(ws: WebSocket, event: str, data: dict):
    """Отправить JSON-сообщение по WebSocket."""
    await ws.send_text(json.dumps({"event": event, **data}))


@app.websocket("/ws/run")
async def websocket_run(ws: WebSocket):
    """
    WebSocket-эндпоинт.
    Клиент подключается → запускаем пайплайн в executor → стримим прогресс.
    """
    await ws.accept()
    loop = asyncio.get_event_loop()

    try:
        # ── Шаг 1: EDA ────────────────────────────────────────────────────────
        await send(ws, "step_start", {"step": 1, "title": "Загрузка и EDA"})
        eda = await loop.run_in_executor(
            None,
            lambda: __import__("main").step1_load_and_eda(str(PROJECT_DIR)),
        )
        await send(ws, "step_done", {
            "step": 1,
            "data": {
                "shape": list(eda["shape"]),
                "plots": {k: f"/plots/{Path(v).name}" for k, v in eda["plots"].items()},
            }
        })

        # ── Шаг 2: Предобработка ──────────────────────────────────────────────
        await send(ws, "step_start", {"step": 2, "title": "Предобработка данных"})
        preprocessed = await loop.run_in_executor(
            None,
            lambda: __import__("main").step2_preprocess(eda["df"]),
        )
        await send(ws, "step_done", {
            "step": 2,
            "data": {
                "train_shape": list(preprocessed["train_shape"]),
                "test_shape": list(preprocessed["test_shape"]),
            }
        })

        # ── Шаг 3: Обучение ───────────────────────────────────────────────────
        await send(ws, "step_start", {"step": 3, "title": "Обучение моделей"})
        models = await loop.run_in_executor(
            None,
            lambda: __import__("main").step3_train_models(
                preprocessed["X_train"], preprocessed["y_train"]
            ),
        )
        await send(ws, "step_done", {
            "step": 3,
            "data": {"models": list(models.keys())}
        })

        # ── Шаг 4: Оценка ─────────────────────────────────────────────────────
        await send(ws, "step_start", {"step": 4, "title": "Оценка и сравнение"})
        evaluation = await loop.run_in_executor(
            None,
            lambda: __import__("main").step4_evaluate(
                models,
                preprocessed["X_test"],
                preprocessed["y_test"],
                str(PROJECT_DIR),
            ),
        )
        eval_plots = {k: f"/plots/{Path(v).name}" for k, v in evaluation["plots"].items()}
        await send(ws, "step_done", {
            "step": 4,
            "data": {
                "results": evaluation["results"],
                "plots": eval_plots,
            }
        })

        # ── Финал ─────────────────────────────────────────────────────────────
        await send(ws, "pipeline_complete", {
            "all_plots": {
                **{k: f"/plots/{Path(v).name}" for k, v in eda["plots"].items()},
                **eval_plots,
            },
            "results": evaluation["results"],
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await send(ws, "error", {"message": str(e)})


# ── Catch-all: раздаём index.html для SPA (production) ────────────────────
@app.get("/{full_path:path}")
async def serve_spa(request: Request, full_path: str):
    """На production раздаёт React SPA. В dev — не мешает."""
    index = PROJECT_DIR / "frontend" / "dist" / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"detail": "Frontend not built. Run: cd frontend && npm run build"}
